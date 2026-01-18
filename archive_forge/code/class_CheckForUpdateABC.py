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
class CheckForUpdateABC(abc.ABC):
    """Abstract class to check for update.

    Args:
        version: Current JupyterLab version

    Attributes:
        version - str: Current JupyterLab version
        logger - logging.Logger: Server logger
    """

    def __init__(self, version: str) -> None:
        self.version = version

    @abc.abstractmethod
    async def __call__(self) -> Awaitable[Union[None, str, Tuple[str, Tuple[str, str]]]]:
        """Get the notification message if a new version is available.

        Returns:
            None if there is not update.
            or the notification message
            or the notification message and a tuple(label, URL link) for the user to get more information
        """
        msg = 'CheckForUpdateABC.__call__ is not implemented'
        raise NotImplementedError(msg)