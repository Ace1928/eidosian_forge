import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.errors import CommError
from wandb.sdk.artifacts.artifact_state import ArtifactState
from wandb.sdk.data_types._dtypes import InvalidType, Type, TypeRegistry
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import (
class Job:
    _name: str
    _input_types: Type
    _output_types: Type
    _entity: str
    _project: str
    _entrypoint: List[str]
    _notebook_job: bool
    _partial: bool

    def __init__(self, api: 'Api', name, path: Optional[str]=None) -> None:
        try:
            self._job_artifact = api.artifact(name, type='job')
        except CommError:
            raise CommError(f'Job artifact {name} not found')
        if path:
            self._fpath = path
            self._job_artifact.download(root=path)
        else:
            self._fpath = self._job_artifact.download()
        self._name = name
        self._api = api
        self._entity = api.default_entity
        with open(os.path.join(self._fpath, 'wandb-job.json')) as f:
            self._job_info: Mapping[str, Any] = json.load(f)
        source_info = self._job_info.get('source', {})
        self._notebook_job = source_info.get('notebook', False)
        self._entrypoint = source_info.get('entrypoint')
        self._args = source_info.get('args')
        self._partial = self._job_info.get('_partial', False)
        self._requirements_file = os.path.join(self._fpath, 'requirements.frozen.txt')
        self._input_types = TypeRegistry.type_from_dict(self._job_info.get('input_types'))
        self._output_types = TypeRegistry.type_from_dict(self._job_info.get('output_types'))
        if self._job_info.get('source_type') == 'artifact':
            self._set_configure_launch_project(self._configure_launch_project_artifact)
        if self._job_info.get('source_type') == 'repo':
            self._set_configure_launch_project(self._configure_launch_project_repo)
        if self._job_info.get('source_type') == 'image':
            self._set_configure_launch_project(self._configure_launch_project_container)

    @property
    def name(self):
        return self._name

    def _set_configure_launch_project(self, func):
        self.configure_launch_project = func

    def _get_code_artifact(self, artifact_string):
        artifact_string, base_url, is_id = util.parse_artifact_string(artifact_string)
        if is_id:
            code_artifact = wandb.Artifact._from_id(artifact_string, self._api._client)
        else:
            code_artifact = self._api.artifact(name=artifact_string, type='code')
        if code_artifact is None:
            raise LaunchError('No code artifact found')
        if code_artifact.state == ArtifactState.DELETED:
            raise LaunchError(f'Job {self.name} references deleted code artifact {code_artifact.name}')
        return code_artifact

    def _configure_launch_project_notebook(self, launch_project):
        new_fname = convert_jupyter_notebook_to_script(self._entrypoint[-1], launch_project.project_dir)
        new_entrypoint = self._entrypoint
        new_entrypoint[-1] = new_fname
        launch_project.set_entry_point(new_entrypoint)

    def _configure_launch_project_repo(self, launch_project):
        git_info = self._job_info.get('source', {}).get('git', {})
        _fetch_git_repo(launch_project.project_dir, git_info['remote'], git_info['commit'])
        if os.path.exists(os.path.join(self._fpath, 'diff.patch')):
            with open(os.path.join(self._fpath, 'diff.patch')) as f:
                apply_patch(f.read(), launch_project.project_dir)
        shutil.copy(self._requirements_file, launch_project.project_dir)
        launch_project.python_version = self._job_info.get('runtime')
        if self._notebook_job:
            self._configure_launch_project_notebook(launch_project)
        else:
            launch_project.set_entry_point(self._entrypoint)

    def _configure_launch_project_artifact(self, launch_project):
        artifact_string = self._job_info.get('source', {}).get('artifact')
        if artifact_string is None:
            raise LaunchError(f'Job {self.name} had no source artifact')
        code_artifact = self._get_code_artifact(artifact_string)
        launch_project.python_version = self._job_info.get('runtime')
        shutil.copy(self._requirements_file, launch_project.project_dir)
        code_artifact.download(launch_project.project_dir)
        if self._notebook_job:
            self._configure_launch_project_notebook(launch_project)
        else:
            launch_project.set_entry_point(self._entrypoint)

    def _configure_launch_project_container(self, launch_project):
        launch_project.docker_image = self._job_info.get('source', {}).get('image')
        if launch_project.docker_image is None:
            raise LaunchError('Job had malformed source dictionary without an image key')
        if self._entrypoint:
            launch_project.set_entry_point(self._entrypoint)

    def set_entrypoint(self, entrypoint: List[str]):
        self._entrypoint = entrypoint

    def call(self, config, project=None, entity=None, queue=None, resource='local-container', resource_args=None, template_variables=None, project_queue=None, priority=None):
        from wandb.sdk.launch import _launch_add
        run_config = {}
        for key, item in config.items():
            if util._is_artifact_object(item):
                if isinstance(item, wandb.Artifact) and item.is_draft():
                    raise ValueError('Cannot queue jobs with unlogged artifacts')
                run_config[key] = util.artifact_to_json(item)
        run_config.update(config)
        assigned_config_type = self._input_types.assign(run_config)
        if self._partial:
            wandb.termwarn("Launching manually created job for the first time, can't verify types")
        elif isinstance(assigned_config_type, InvalidType):
            raise TypeError(self._input_types.explain(run_config))
        queued_run = _launch_add.launch_add(job=self._name, config={'overrides': {'run_config': run_config}}, template_variables=template_variables, project=project or self._project, entity=entity or self._entity, queue_name=queue, resource=resource, project_queue=project_queue, resource_args=resource_args, priority=priority)
        return queued_run