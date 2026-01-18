import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union
from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders
class ClientConnectorError(ClientOSError):
    """Client connector error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        a connection can not be established.
    """

    def __init__(self, connection_key: ConnectionKey, os_error: OSError) -> None:
        self._conn_key = connection_key
        self._os_error = os_error
        super().__init__(os_error.errno, os_error.strerror)
        self.args = (connection_key, os_error)

    @property
    def os_error(self) -> OSError:
        return self._os_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> Union[SSLContext, bool, 'Fingerprint']:
        return self._conn_key.ssl

    def __str__(self) -> str:
        return 'Cannot connect to host {0.host}:{0.port} ssl:{1} [{2}]'.format(self, 'default' if self.ssl is True else self.ssl, self.strerror)
    __reduce__ = BaseException.__reduce__