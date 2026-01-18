import enum
import logging
import os
import tempfile
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
import wandb
import wandb.docker as docker
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch import utils
from wandb.sdk.lib.runid import generate_id
from .errors import LaunchError
from .utils import LOG_PREFIX, recursive_macro_sub
def _fetch_project_local(self, internal_api: Api) -> None:
    """Fetch a project (either wandb run or git repo) into a local directory, returning the path to the local project directory."""
    assert self.source != LaunchSource.LOCAL and self.source != LaunchSource.JOB
    assert isinstance(self.uri, str)
    assert self.project_dir is not None
    _logger.info('Fetching project locally...')
    if utils._is_wandb_uri(self.uri):
        source_entity, source_project, source_run_name = utils.parse_wandb_uri(self.uri)
        run_info = utils.fetch_wandb_project_run_info(source_entity, source_project, source_run_name, internal_api)
        program_name = run_info.get('codePath') or run_info['program']
        self.python_version = run_info.get('python', '3')
        downloaded_code_artifact = utils.check_and_download_code_artifacts(source_entity, source_project, source_run_name, internal_api, self.project_dir)
        if not downloaded_code_artifact:
            if not run_info['git']:
                raise LaunchError('Reproducing a run requires either an associated git repo or a code artifact logged with `run.log_code()`')
            branch_name = utils._fetch_git_repo(self.project_dir, run_info['git']['remote'], run_info['git']['commit'])
            if self.git_version is None:
                self.git_version = branch_name
            patch = utils.fetch_project_diff(source_entity, source_project, source_run_name, internal_api)
            if patch:
                utils.apply_patch(patch, self.project_dir)
            if not os.path.exists(os.path.join(self.project_dir, program_name)):
                downloaded_entrypoint = utils.download_entry_point(source_entity, source_project, source_run_name, internal_api, program_name, self.project_dir)
                if not downloaded_entrypoint:
                    raise LaunchError(f'Entrypoint file: {program_name} does not exist, and could not be downloaded. Please specify the entrypoint for this run.')
        if '_session_history.ipynb' in os.listdir(self.project_dir) or '.ipynb' in program_name:
            program_name = utils.convert_jupyter_notebook_to_script(program_name, self.project_dir)
        utils.download_wandb_python_deps(source_entity, source_project, source_run_name, internal_api, self.project_dir)
        if not self._entry_point:
            _, ext = os.path.splitext(program_name)
            if ext == '.py':
                entry_point = ['python', program_name]
            elif ext == '.sh':
                command = os.environ.get('SHELL', 'bash')
                entry_point = [command, program_name]
            else:
                raise LaunchError(f'Unsupported entrypoint: {program_name}')
            self.set_entry_point(entry_point)
        if not self.override_args:
            self.override_args = run_info['args']
    else:
        assert utils._GIT_URI_REGEX.match(self.uri), 'Non-wandb URI %s should be a Git URI' % self.uri
        if not self._entry_point:
            wandb.termlog(f'{LOG_PREFIX}Entry point for repo not specified, defaulting to python main.py')
            self.set_entry_point(EntrypointDefaults.PYTHON)
        branch_name = utils._fetch_git_repo(self.project_dir, self.uri, self.git_version)
        if self.git_version is None:
            self.git_version = branch_name