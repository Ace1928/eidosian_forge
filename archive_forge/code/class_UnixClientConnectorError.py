import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class UnixClientConnectorError(ClientConnectorError):
    """Unix connector error.

    Raised in :py:class:`aiohttp.connector.UnixConnector`
    if connection to unix socket can not be established.
    """

    def __init__(self, path: str, connection_key: ConnectionKey, os_error: OSError) -> None:
        self._path = path
        super().__init__(connection_key, os_error)

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return 'Cannot connect to unix socket {0.path} ssl:{1} [{2}]'.format(self, 'default' if self.ssl is True else self.ssl, self.strerror)