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