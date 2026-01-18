import logging
import os
from typing import Dict, Optional
from wandb.sdk.launch.errors import LaunchError
from wandb.util import get_module
from ..utils import S3_URI_RE, event_loop_thread_exec
from .abstract import AbstractEnvironment
Verify that s3 storage is configured correctly.

        This will check that the bucket exists and that the credentials are
        configured correctly.

        Arguments:
            uri (str): The URI of the storage.

        Raises:
            LaunchError: If the storage is not configured correctly or the URI is
                not a valid s3 URI.

        Returns:
            None
        