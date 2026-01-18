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
class KeyDBSSLContext:
    __slots__ = ('keyfile', 'certfile', 'cert_reqs', 'ca_certs', 'ca_data', 'context', 'check_hostname')

    def __init__(self, keyfile: Optional[str]=None, certfile: Optional[str]=None, cert_reqs: Optional[str]=None, ca_certs: Optional[str]=None, ca_data: Optional[str]=None, check_hostname: bool=False):
        self.keyfile = keyfile
        self.certfile = certfile
        if cert_reqs is None:
            self.cert_reqs = ssl.CERT_NONE
        elif isinstance(cert_reqs, str):
            CERT_REQS = {'none': ssl.CERT_NONE, 'optional': ssl.CERT_OPTIONAL, 'required': ssl.CERT_REQUIRED}
            if cert_reqs not in CERT_REQS:
                raise KeyDBError(f'Invalid SSL Certificate Requirements Flag: {cert_reqs}')
            self.cert_reqs = CERT_REQS[cert_reqs]
        self.ca_certs = ca_certs
        self.ca_data = ca_data
        self.check_hostname = check_hostname
        self.context: Optional[ssl.SSLContext] = None

    def get(self) -> ssl.SSLContext:
        if not self.context:
            context = ssl.create_default_context()
            context.check_hostname = self.check_hostname
            context.verify_mode = self.cert_reqs
            if self.certfile and self.keyfile:
                context.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
            if self.ca_certs or self.ca_data:
                context.load_verify_locations(cafile=self.ca_certs, cadata=self.ca_data)
            self.context = context
        return self.context