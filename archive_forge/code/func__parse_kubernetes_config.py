import json
import logging
import os
import yaml
import mlflow.projects.databricks
import mlflow.utils.uri
from mlflow import tracking
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException, MlflowException
from mlflow.projects.backend import loader
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects.utils import (
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.mlflow_tags import (
def _parse_kubernetes_config(backend_config):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    if not backend_config:
        raise ExecutionException('Backend_config file not found.')
    kube_config = backend_config.copy()
    if 'kube-job-template-path' not in backend_config.keys():
        raise ExecutionException("'kube-job-template-path' attribute must be specified in backend_config.")
    kube_job_template = backend_config['kube-job-template-path']
    if os.path.exists(kube_job_template):
        with open(kube_job_template) as job_template:
            yaml_obj = yaml.safe_load(job_template.read())
        kube_job_template = yaml_obj
        kube_config['kube-job-template'] = kube_job_template
    else:
        raise ExecutionException(f"Could not find 'kube-job-template-path': {kube_job_template}")
    if 'kube-context' not in backend_config.keys():
        _logger.debug('Could not find kube-context in backend_config. Using current context or in-cluster config.')
    if 'repository-uri' not in backend_config.keys():
        raise ExecutionException("Could not find 'repository-uri' in backend_config.")
    return kube_config