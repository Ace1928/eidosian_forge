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
def _create_artifact_metadata(path: str, entrypoint: str, runtime: Optional[str]=None) -> Tuple[Dict[str, Any], List[str]]:
    if not os.path.isdir(path):
        wandb.termerror('Path must be a valid file or directory')
        return ({}, [])
    requirements = []
    depspath = os.path.join(path, 'requirements.txt')
    if os.path.exists(depspath):
        with open(depspath) as f:
            requirements = f.read().splitlines()
    if runtime:
        python_version = _clean_python_version(runtime)
    else:
        python_version = '.'.join(get_current_python_version())
    metadata = {'python': python_version, 'codePath': entrypoint}
    return (metadata, requirements)