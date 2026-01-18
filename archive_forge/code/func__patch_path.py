import importlib
import json
import os
from pathlib import Path
import re
import sys
import typer
from typing import Optional
import uuid
import yaml
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.resources import resources_to_json, json_to_resources
from ray.tune.tune import run_experiments
from ray.tune.schedulers import create_scheduler
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import FrameworkEnum, SupportedFileType
from ray.rllib.common import download_example_file, get_file_type
def _patch_path(path: str):
    """
    Patch a path to be relative to the current working directory.

    Args:
        path: relative input path.

    Returns: Patched path.
    """
    rllib_dir = Path(__file__).parent
    if isinstance(path, list):
        return [_patch_path(i) for i in path]
    elif isinstance(path, dict):
        return {_patch_path(k): _patch_path(v) for k, v in path.items()}
    elif isinstance(path, str):
        if os.path.exists(path):
            return path
        else:
            abs_path = str(rllib_dir.absolute().joinpath(path))
            return abs_path if os.path.exists(abs_path) else path
    else:
        return path