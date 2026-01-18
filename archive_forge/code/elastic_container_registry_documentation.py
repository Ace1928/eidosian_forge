import base64
import logging
from typing import Dict, Optional, Tuple
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from wandb.sdk.launch.utils import (
from wandb.util import get_module
Check if the image tag exists.

        Arguments:
            image_uri (str): The full image_uri.

        Returns:
            bool: True if the image tag exists.
        