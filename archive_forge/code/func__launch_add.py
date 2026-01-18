import asyncio
import pprint
from typing import Any, Dict, List, Optional, Union
import wandb
import wandb.apis.public as public
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.builder.build import build_image_from_project
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
from ._project_spec import LaunchProject
def _launch_add(api: Api, uri: Optional[str], job: Optional[str], config: Optional[Dict[str, Any]], template_variables: Optional[dict], project: Optional[str], entity: Optional[str], queue_name: Optional[str], resource: Optional[str], entry_point: Optional[List[str]], name: Optional[str], version: Optional[str], docker_image: Optional[str], project_queue: Optional[str], resource_args: Optional[Dict[str, Any]]=None, run_id: Optional[str]=None, build: Optional[bool]=False, repository: Optional[str]=None, sweep_id: Optional[str]=None, author: Optional[str]=None, priority: Optional[int]=None) -> 'public.QueuedRun':
    launch_spec = construct_launch_spec(uri, job, api, name, project, entity, docker_image, resource, entry_point, version, resource_args, config, run_id, repository, author, sweep_id)
    if build:
        if resource == 'local-process':
            raise LaunchError('Cannot build a docker image for the resource: local-process')
        if launch_spec.get('job') is not None:
            wandb.termwarn("Build doesn't support setting a job. Overwriting job.")
            launch_spec['job'] = None
        launch_project = LaunchProject.from_spec(launch_spec, api)
        docker_image_uri = asyncio.run(build_image_from_project(launch_project, api, config or {}))
        run = wandb.run or wandb.init(project=launch_spec['project'], entity=launch_spec['entity'], job_type='launch_job')
        job_artifact = run._log_job_artifact_with_image(docker_image_uri, launch_project.override_args)
        job_name = job_artifact.wait().name
        job = f'{launch_spec['entity']}/{launch_spec['project']}/{job_name}'
        launch_spec['job'] = job
        launch_spec['uri'] = None
    if queue_name is None:
        queue_name = 'default'
    if project_queue is None:
        project_queue = LAUNCH_DEFAULT_PROJECT
    spec_template_vars = launch_spec.get('template_variables')
    if isinstance(spec_template_vars, dict):
        launch_spec.pop('template_variables')
        if template_variables is None:
            template_variables = spec_template_vars
        else:
            template_variables = {**spec_template_vars, **template_variables}
    validate_launch_spec_source(launch_spec)
    res = push_to_queue(api, queue_name, launch_spec, template_variables, project_queue, priority)
    if res is None or 'runQueueItemId' not in res:
        raise LaunchError('Error adding run to queue')
    updated_spec = res.get('runSpec')
    if updated_spec:
        if updated_spec.get('resource_args'):
            launch_spec['resource_args'] = updated_spec.get('resource_args')
        if updated_spec.get('resource'):
            launch_spec['resource'] = updated_spec.get('resource')
    if project_queue == LAUNCH_DEFAULT_PROJECT:
        wandb.termlog(f'{LOG_PREFIX}Added run to queue {queue_name}.')
    else:
        wandb.termlog(f'{LOG_PREFIX}Added run to queue {project_queue}/{queue_name}.')
    wandb.termlog(f'{LOG_PREFIX}Launch spec:\n{pprint.pformat(launch_spec)}\n')
    public_api = public.Api()
    if job is not None:
        try:
            public_api.artifact(job, type='job')
        except (ValueError, CommError) as e:
            raise LaunchError(f'Unable to fetch job with name {job}: {e}')
    queued_run = public_api.queued_run(launch_spec['entity'], launch_spec['project'], queue_name, res['runQueueItemId'], project_queue, priority)
    return queued_run