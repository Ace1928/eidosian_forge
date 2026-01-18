import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def _inject_wandb_config_env_vars(config: Dict[str, Any], env_dict: Dict[str, Any], maximum_env_length: int) -> None:
    str_config = json.dumps(config)
    if len(str_config) <= maximum_env_length:
        env_dict['WANDB_CONFIG'] = str_config
        return
    chunks = [str_config[i:i + maximum_env_length] for i in range(0, len(str_config), maximum_env_length)]
    config_chunks_dict = {f'WANDB_CONFIG_{i}': chunk for i, chunk in enumerate(chunks)}
    env_dict.update(config_chunks_dict)