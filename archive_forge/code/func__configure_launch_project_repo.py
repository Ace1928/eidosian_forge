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