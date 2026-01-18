import abc
import hashlib
import json
import xml.etree.ElementTree as ET  # noqa
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Optional, Tuple, Union
from jupyter_server.base.handlers import APIHandler
from jupyterlab_server.translation_utils import translator
from packaging.version import parse
from tornado import httpclient, web
from jupyterlab._version import __version__
class NeverCheckForUpdate(CheckForUpdateABC):
    """Check update version that does nothing.

    This is provided for administrators that want to
    turn off requesting external resources.

    Args:
        version: Current JupyterLab version

    Attributes:
        version - str: Current JupyterLab version
        logger - logging.Logger: Server logger
    """

    async def __call__(self) -> Awaitable[None]:
        """Get the notification message if a new version is available.

        Returns:
            None if there is no update.
            or the notification message
            or the notification message and a tuple(label, URL link) for the user to get more information
        """
        return None