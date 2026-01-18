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
def _bake_output(registry, accept_header, accept_encoding_header, params, disable_compression):
    """Bake output for metrics output."""
    encoder, content_type = choose_encoder(accept_header)
    if 'name[]' in params:
        registry = registry.restricted_registry(params['name[]'])
    output = encoder(registry)
    headers = [('Content-Type', content_type)]
    if not disable_compression and gzip_accepted(accept_encoding_header):
        output = gzip.compress(output)
        headers.append(('Content-Encoding', 'gzip'))
    return ('200 OK', headers, output)