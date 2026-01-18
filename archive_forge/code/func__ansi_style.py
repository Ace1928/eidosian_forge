from __future__ import annotations
import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit
from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri
def _ansi_style(value: str, *styles: str) -> str:
    if not _log_add_style:
        return value
    codes = {'bold': 1, 'red': 31, 'green': 32, 'yellow': 33, 'magenta': 35, 'cyan': 36}
    for style in styles:
        value = f'\x1b[{codes[style]}m{value}'
    return f'{value}\x1b[0m'