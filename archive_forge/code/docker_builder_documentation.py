import logging
import os
from typing import Any, Dict, Optional
import wandb
import wandb.docker as docker
from wandb.sdk.launch.agent.job_status_tracker import JobAndRunStatusTracker
from wandb.sdk.launch.builder.abstract import AbstractBuilder
from wandb.sdk.launch.builder.build import registry_from_uri
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from .._project_spec import EntryPoint, LaunchProject
from ..errors import LaunchDockerError, LaunchError
from ..registry.anon import AnonynmousRegistry
from ..registry.local_registry import LocalRegistry
from ..utils import (
from .build import (
Build the image for the given project.

        Arguments:
            launch_project (LaunchProject): The project to build.
            entrypoint (EntryPoint): The entrypoint to use.
        