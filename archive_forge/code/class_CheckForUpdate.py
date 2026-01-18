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
class CheckForUpdate(CheckForUpdateABC):
    """Default class to check for update.

    Args:
        version: Current JupyterLab version

    Attributes:
        version - str: Current JupyterLab version
        logger - logging.Logger: Server logger
    """

    async def __call__(self) -> Awaitable[Tuple[str, Tuple[str, str]]]:
        """Get the notification message if a new version is available.

        Returns:
            None if there is no update.
            or the notification message
            or the notification message and a tuple(label, URL link) for the user to get more information
        """
        http_client = httpclient.AsyncHTTPClient()
        try:
            response = await http_client.fetch(JUPYTERLAB_LAST_RELEASE_URL, headers={'Content-Type': 'application/json'})
            data = json.loads(response.body).get('info')
            last_version = data['version']
        except Exception as e:
            self.logger.debug('Failed to get latest version', exc_info=e)
            return None
        else:
            if parse(self.version) < parse(last_version):
                trans = translator.load('jupyterlab')
                return (trans.__(f'A newer version ({last_version}) of JupyterLab is available.'), (trans.__('Open changelog'), f'{JUPYTERLAB_RELEASE_URL}{last_version}'))
            else:
                return None