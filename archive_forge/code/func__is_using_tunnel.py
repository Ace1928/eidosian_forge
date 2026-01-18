from __future__ import absolute_import
import datetime
import logging
import os
import re
import socket
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401
from .util.proxy import create_proxy_ssl_context
from ._collections import HTTPHeaderDict  # noqa (historical, removed in v2)
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
def _is_using_tunnel(self):
    return getattr(self, '_tunnel_host', None)