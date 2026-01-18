import hashlib
import json
import logging
import os
import posixpath
import re
import tempfile
import textwrap
import time
from shlex import quote
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import MLFLOW_LOCAL_BACKEND_RUN_ID_CONFIG
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import databricks_utils, file_utils, rest_utils
from mlflow.utils.mlflow_tags import (
from mlflow.utils.uri import is_databricks_uri, is_http_uri
from mlflow.version import VERSION, is_release_version
class DatabricksJobRunner:
    """
    Helper class for running an MLflow project as a Databricks Job.

    Args:
        databricks_profile: Optional Databricks CLI profile to use to fetch hostname &
           authentication information when making Databricks API requests.
    """

    def __init__(self, databricks_profile_uri):
        self.databricks_profile_uri = databricks_profile_uri

    def _databricks_api_request(self, endpoint, method, **kwargs):
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile_uri)
        return rest_utils.http_request_safe(host_creds=host_creds, endpoint=endpoint, method=method, **kwargs)

    def _jobs_runs_submit(self, req_body):
        response = self._databricks_api_request(endpoint='/api/2.0/jobs/runs/submit', method='POST', json=req_body)
        return json.loads(response.text)

    def _upload_to_dbfs(self, src_path, dbfs_fuse_uri):
        """
        Upload the file at `src_path` to the specified DBFS URI within the Databricks workspace
        corresponding to the default Databricks CLI profile.
        """
        _logger.info('=== Uploading project to DBFS path %s ===', dbfs_fuse_uri)
        http_endpoint = dbfs_fuse_uri
        with open(src_path, 'rb') as f:
            try:
                self._databricks_api_request(endpoint=http_endpoint, method='POST', data=f)
            except MlflowException as e:
                if 'Error 409' in e.message and 'File already exists' in e.message:
                    _logger.info('=== Did not overwrite existing DBFS path %s ===', dbfs_fuse_uri)
                else:
                    raise e

    def _dbfs_path_exists(self, dbfs_path):
        """
        Return True if the passed-in path exists in DBFS for the workspace corresponding to the
        default Databricks CLI profile. The path is expected to be a relative path to the DBFS root
        directory, e.g. 'path/to/file'.
        """
        host_creds = databricks_utils.get_databricks_host_creds(self.databricks_profile_uri)
        response = rest_utils.http_request(host_creds=host_creds, endpoint='/api/2.0/dbfs/get-status', method='GET', json={'path': f'/{dbfs_path}'})
        try:
            json_response_obj = json.loads(response.text)
        except Exception:
            raise MlflowException(f'API request to check existence of file at DBFS path {dbfs_path} failed with status code {response.status_code}. Response body: {response.text}')
        error_code_field = 'error_code'
        if error_code_field in json_response_obj:
            if json_response_obj[error_code_field] == 'RESOURCE_DOES_NOT_EXIST':
                return False
            raise ExecutionException(f'Got unexpected error response when checking whether file {dbfs_path} exists in DBFS: {json_response_obj}')
        return True

    def _upload_project_to_dbfs(self, project_dir, experiment_id):
        """
        Tars a project directory into an archive in a temp dir and uploads it to DBFS, returning
        the HDFS-style URI of the tarball in DBFS (e.g. dbfs:/path/to/tar).

        Args:
            project_dir: Path to a directory containing an MLflow project to upload to DBFS (e.g.
                a directory containing an MLproject file).
        """
        with tempfile.TemporaryDirectory() as temp_tarfile_dir:
            temp_tar_filename = os.path.join(temp_tarfile_dir, 'project.tar.gz')

            def custom_filter(x):
                return None if os.path.basename(x.name) == 'mlruns' else x
            directory_size = file_utils._get_local_project_dir_size(project_dir)
            _logger.info(f'=== Creating tarball from {project_dir} in temp directory {temp_tarfile_dir} ===')
            _logger.info(f'=== Total file size to compress: {directory_size} KB ===')
            file_utils.make_tarfile(temp_tar_filename, project_dir, DB_TARFILE_ARCHIVE_NAME, custom_filter=custom_filter)
            with open(temp_tar_filename, 'rb') as tarred_project:
                tarfile_hash = hashlib.sha256(tarred_project.read()).hexdigest()
            dbfs_path = posixpath.join(DBFS_EXPERIMENT_DIR_BASE, str(experiment_id), 'projects-code', f'{tarfile_hash}.tar.gz')
            tar_size = file_utils._get_local_file_size(temp_tar_filename)
            dbfs_fuse_uri = posixpath.join('/dbfs', dbfs_path)
            if not self._dbfs_path_exists(dbfs_path):
                _logger.info(f'=== Uploading project tarball (size: {tar_size} KB) to {dbfs_fuse_uri} ===')
                self._upload_to_dbfs(temp_tar_filename, dbfs_fuse_uri)
                _logger.info('=== Finished uploading project to %s ===', dbfs_fuse_uri)
            else:
                _logger.info('=== Project already exists in DBFS ===')
        return dbfs_fuse_uri

    def _run_shell_command_job(self, project_uri, command, env_vars, cluster_spec):
        """
        Run the specified shell command on a Databricks cluster.

        Args:
            project_uri: URI of the project from which the shell command originates.
            command: Shell command to run.
            env_vars: Environment variables to set in the process running ``command``.
            cluster_spec: Dictionary containing a `Databricks cluster specification
                <https://docs.databricks.com/dev-tools/api/latest/jobs.html#clusterspec>`_
                or a `Databricks new cluster specification
                <https://docs.databricks.com/dev-tools/api/latest/jobs.html#jobsclusterspecnewcluster>`_
                to use when launching a run. If you specify libraries, this function
                will add MLflow to the library list. This function does not support
                installation of conda environment libraries on the workers.

        Returns:
            ID of the Databricks job run. Can be used to query the run's status via the
            Databricks `Runs Get <https://docs.databricks.com/api/latest/jobs.html#runs-get>`_ API.
        """
        if is_release_version():
            mlflow_lib = {'pypi': {'package': f'mlflow=={VERSION}'}}
        else:
            _logger.warning('Your client is running a non-release version of MLflow. This version is not available on the databricks runtime. MLflow will fallback the MLflow version provided by the runtime. This might lead to unforeseen issues. ')
            mlflow_lib = {'pypi': {'package': f"'mlflow<={VERSION}'"}}
        if 'new_cluster' in cluster_spec:
            cluster_spec_libraries = cluster_spec.get('libraries', [])
            libraries = cluster_spec_libraries if _contains_mlflow_git_uri(cluster_spec_libraries) else cluster_spec_libraries + [mlflow_lib]
            cluster_spec = cluster_spec['new_cluster']
        else:
            libraries = [mlflow_lib]
        req_body_json = {'run_name': f'MLflow Run for {project_uri}', 'new_cluster': cluster_spec, 'shell_command_task': {'command': command, 'env_vars': env_vars}, 'libraries': libraries}
        _logger.info('=== Submitting a run to execute the MLflow project... ===')
        run_submit_res = self._jobs_runs_submit(req_body_json)
        return run_submit_res['run_id']

    def run_databricks(self, uri, entry_point, work_dir, parameters, experiment_id, cluster_spec, run_id, env_manager):
        tracking_uri = _get_tracking_uri_for_run()
        dbfs_fuse_uri = self._upload_project_to_dbfs(work_dir, experiment_id)
        env_vars = {MLFLOW_TRACKING_URI.name: tracking_uri, MLFLOW_EXPERIMENT_ID.name: experiment_id}
        _logger.info('=== Running entry point %s of project %s on Databricks ===', entry_point, uri)
        command = _get_databricks_run_cmd(dbfs_fuse_uri, run_id, entry_point, parameters, env_manager)
        return self._run_shell_command_job(uri, command, env_vars, cluster_spec)

    def _get_status(self, databricks_run_id):
        run_state = self.get_run_result_state(databricks_run_id)
        if run_state is None:
            return RunStatus.RUNNING
        if run_state == 'SUCCESS':
            return RunStatus.FINISHED
        return RunStatus.FAILED

    def get_status(self, databricks_run_id):
        return RunStatus.to_string(self._get_status(databricks_run_id))

    def get_run_result_state(self, databricks_run_id):
        """
        Get the run result state (string) of a Databricks job run.

        Args:
            databricks_run_id: Integer Databricks job run ID.

        Returns:
            `RunResultState <https://docs.databricks.com/api/latest/jobs.html#runresultstate>`_ or
            None if the run is still active.
        """
        res = self.jobs_runs_get(databricks_run_id)
        return res['state'].get('result_state', None)

    def jobs_runs_cancel(self, databricks_run_id):
        response = self._databricks_api_request(endpoint='/api/2.0/jobs/runs/cancel', method='POST', json={'run_id': databricks_run_id})
        return json.loads(response.text)

    def jobs_runs_get(self, databricks_run_id):
        response = self._databricks_api_request(endpoint='/api/2.0/jobs/runs/get', method='GET', params={'run_id': databricks_run_id})
        return json.loads(response.text)