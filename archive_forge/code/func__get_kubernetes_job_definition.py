import logging
import os
import time
from datetime import datetime
from shlex import quote, split
from threading import RLock
import kubernetes
from kubernetes.config.config_exception import ConfigException
import docker
from mlflow.entities import RunStatus
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
def _get_kubernetes_job_definition(project_name, image_tag, image_digest, command, env_vars, job_template):
    container_image = image_tag + '@' + image_digest
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = f'{project_name}-{timestamp}'
    _logger.info('=== Creating Job %s ===', job_name)
    if os.environ.get('KUBE_MLFLOW_TRACKING_URI') is not None:
        env_vars['MLFLOW_TRACKING_URI'] = os.environ['KUBE_MLFLOW_TRACKING_URI']
    environment_variables = [{'name': k, 'value': v} for k, v in env_vars.items()]
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = project_name
    job_template['spec']['template']['spec']['containers'][0]['image'] = container_image
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    if 'env' not in job_template['spec']['template']['spec']['containers'][0].keys():
        job_template['spec']['template']['spec']['containers'][0]['env'] = []
    job_template['spec']['template']['spec']['containers'][0]['env'] += environment_variables
    return job_template