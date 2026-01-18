import uuid
import logging
import asyncio
import copy
import enum
import errno
import inspect
import io
import os
import socket
import ssl
import threading
import weakref
from itertools import chain
from types import MappingProxyType
from typing import (
from urllib.parse import ParseResult, parse_qs, unquote, urlparse
import async_timeout
from aiokeydb.v1.backoff import NoBackoff
from aiokeydb.v1.asyncio.retry import Retry
from aiokeydb.v1.compat import Protocol, TypedDict
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.credentials import CredentialProvider, UsernamePasswordCredentialProvider
from aiokeydb.v1.typing import EncodableT, EncodedT
from aiokeydb.v1.utils import HIREDIS_AVAILABLE, str_if_bytes, set_ulimits
class AsyncSSLConnection(AsyncConnection):

    def __init__(self, ssl_keyfile: Optional[str]=None, ssl_certfile: Optional[str]=None, ssl_cert_reqs: str='required', ssl_ca_certs: Optional[str]=None, ssl_ca_data: Optional[str]=None, ssl_check_hostname: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.ssl_context: KeyDBSSLContext = KeyDBSSLContext(keyfile=ssl_keyfile, certfile=ssl_certfile, cert_reqs=ssl_cert_reqs, ca_certs=ssl_ca_certs, ca_data=ssl_ca_data, check_hostname=ssl_check_hostname)

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