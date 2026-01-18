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
def _configure_launch_project_notebook(self, launch_project):
    new_fname = convert_jupyter_notebook_to_script(self._entrypoint[-1], launch_project.project_dir)
    new_entrypoint = self._entrypoint
    new_entrypoint[-1] = new_fname
    launch_project.set_entry_point(new_entrypoint)