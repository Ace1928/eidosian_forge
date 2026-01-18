import asyncio
import copy
import enum
import inspect
import socket
import ssl
import sys
import warnings
import weakref
from abc import abstractmethod
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.compat import Protocol, TypedDict
from redis.connection import DEFAULT_RESP_VERSION
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from redis.exceptions import (
from redis.typing import EncodableT
from redis.utils import HIREDIS_AVAILABLE, get_lib_version, str_if_bytes
from .._parsers import (
class SSLConnection(Connection):
    """Manages SSL connections to and from the Redis server(s).
    This class extends the Connection class, adding SSL functionality, and making
    use of ssl.SSLContext (https://docs.python.org/3/library/ssl.html#ssl.SSLContext)
    """

    def __init__(self, ssl_keyfile: Optional[str]=None, ssl_certfile: Optional[str]=None, ssl_cert_reqs: str='required', ssl_ca_certs: Optional[str]=None, ssl_ca_data: Optional[str]=None, ssl_check_hostname: bool=False, ssl_min_version: Optional[ssl.TLSVersion]=None, **kwargs):
        self.ssl_context: RedisSSLContext = RedisSSLContext(keyfile=ssl_keyfile, certfile=ssl_certfile, cert_reqs=ssl_cert_reqs, ca_certs=ssl_ca_certs, ca_data=ssl_ca_data, check_hostname=ssl_check_hostname, min_version=ssl_min_version)
        super().__init__(**kwargs)

    def _connection_arguments(self) -> Mapping:
        kwargs = super()._connection_arguments()
        kwargs['ssl'] = self.ssl_context.get()
        return kwargs

    @property
    def keyfile(self):
        return self.ssl_context.keyfile

    @property
    def certfile(self):
        return self.ssl_context.certfile

    @property
    def cert_reqs(self):
        return self.ssl_context.cert_reqs

    @property
    def ca_certs(self):
        return self.ssl_context.ca_certs

    @property
    def ca_data(self):
        return self.ssl_context.ca_data

    @property
    def check_hostname(self):
        return self.ssl_context.check_hostname

    @property
    def min_version(self):
        return self.ssl_context.min_version