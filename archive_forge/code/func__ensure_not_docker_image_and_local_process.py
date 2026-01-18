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
def _ensure_not_docker_image_and_local_process(self) -> None:
    """Ensure that docker image is not specified with local-process resource runner.

        Raises:
            LaunchError: If docker image is specified with local-process resource runner.
        """
    if self.docker_image is not None and self.resource == 'local-process':
        raise LaunchError('Cannot specify docker image with local-process resource runner')