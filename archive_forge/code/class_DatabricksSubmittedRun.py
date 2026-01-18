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
class DatabricksSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Databricks Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.

    Args:
        databricks_run_id: Run ID of the launched Databricks Job.
        mlflow_run_id: ID of the MLflow project run.
        databricks_job_runner: Instance of ``DatabricksJobRunner`` used to make Databricks API
            requests.
    """
    POLL_STATUS_INTERVAL = 30

    def __init__(self, databricks_run_id, mlflow_run_id, databricks_job_runner):
        super().__init__()
        self._databricks_run_id = databricks_run_id
        self._mlflow_run_id = mlflow_run_id
        self._job_runner = databricks_job_runner

    def _print_description_and_log_tags(self):
        _logger.info('=== Launched MLflow run as Databricks job run with ID %s. Getting run status page URL... ===', self._databricks_run_id)
        run_info = self._job_runner.jobs_runs_get(self._databricks_run_id)
        jobs_page_url = run_info['run_page_url']
        _logger.info("=== Check the run's status at %s ===", jobs_page_url)
        host_creds = databricks_utils.get_databricks_host_creds(self._job_runner.databricks_profile_uri)
        tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_RUN_URL, jobs_page_url)
        tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, self._databricks_run_id)
        tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_WEBAPP_URL, host_creds.host)
        job_id = run_info.get('job_id')
        if job_id is not None:
            tracking.MlflowClient().set_tag(self._mlflow_run_id, MLFLOW_DATABRICKS_SHELL_JOB_ID, job_id)

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
        while result_state is None:
            time.sleep(self.POLL_STATUS_INTERVAL)
            result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
        return result_state == 'SUCCESS'

    def cancel(self):
        self._job_runner.jobs_runs_cancel(self._databricks_run_id)
        self.wait()

    def get_status(self):
        return self._job_runner.get_status(self._databricks_run_id)