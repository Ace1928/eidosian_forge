import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import wandb
from wandb.apis.internal import Api
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.launch.builder.build import get_current_python_version
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.utils import _is_git_uri
from wandb.sdk.lib import filesystem
from wandb.util import make_artifact_name_safe
def _handle_artifact_entrypoint(path: str, entrypoint: Optional[str]=None) -> Tuple[str, Optional[str]]:
    if os.path.isfile(path):
        if entrypoint and path.endswith(entrypoint):
            path = path.replace(entrypoint, '')
            wandb.termwarn(f'Both entrypoint provided and path contains file. Using provided entrypoint: {entrypoint}, path is now: {path}')
        elif entrypoint:
            wandb.termwarn(f"Ignoring passed in entrypoint as it does not match file path found in 'path'. Path entrypoint: {path.split('/')[-1]}")
        entrypoint = path.split('/')[-1]
        path = '/'.join(path.split('/')[:-1])
    elif not entrypoint:
        wandb.termerror('Entrypoint not valid')
        return ('', None)
    path = path or '.'
    if not os.path.exists(os.path.join(path, entrypoint)):
        wandb.termerror(f'Could not find execution point: {os.path.join(path, entrypoint)}')
        return ('', None)
    return (path, entrypoint)