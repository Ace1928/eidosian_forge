import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import (
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
from .asgi import make_asgi_app  # noqa
def _use_gateway(method: str, gateway: str, job: str, registry: Optional[CollectorRegistry], grouping_key: Optional[Dict[str, Any]], timeout: Optional[float], handler: Callable) -> None:
    gateway_url = urlparse(gateway)
    if not gateway_url.scheme or gateway_url.scheme not in ['http', 'https']:
        gateway = f'http://{gateway}'
    gateway = gateway.rstrip('/')
    url = '{}/metrics/{}/{}'.format(gateway, *_escape_grouping_key('job', job))
    data = b''
    if method != 'DELETE':
        if registry is None:
            registry = REGISTRY
        data = generate_latest(registry)
    if grouping_key is None:
        grouping_key = {}
    url += ''.join(('/{}/{}'.format(*_escape_grouping_key(str(k), str(v))) for k, v in sorted(grouping_key.items())))
    handler(url=url, method=method, timeout=timeout, headers=[('Content-Type', CONTENT_TYPE_LATEST)], data=data)()