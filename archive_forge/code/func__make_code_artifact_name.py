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
def _make_code_artifact_name(path: str, name: Optional[str]) -> str:
    """Make a code artifact name from a path and user provided name."""
    if name:
        return f'code-{name}'
    clean_path = path.replace('./', '')
    if clean_path[0] == '/':
        clean_path = clean_path[1:]
    if clean_path[-1] == '/':
        clean_path = clean_path[:-1]
    path_name = f'code-{make_artifact_name_safe(clean_path)}'
    return path_name