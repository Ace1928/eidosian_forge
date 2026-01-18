import re
from typing import TYPE_CHECKING, Optional, Tuple
from wandb.sdk.launch.environment.azure_environment import AzureEnvironment
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.utils import AZURE_CONTAINER_REGISTRY_URI_REGEX
from wandb.util import get_module
from .abstract import AbstractRegistry
Check if image exists in container registry.

        Args:
            image_uri (str): Image URI to check.

        Returns:
            bool: True if image exists, False otherwise.
        