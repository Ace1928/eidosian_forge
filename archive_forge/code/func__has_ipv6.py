from __future__ import print_function
import logging
import os
import socket
import ssl
import sys
import threading
import warnings
from datetime import datetime
import tornado.httpserver
import tornado.ioloop
import tornado.netutil
import tornado.web
import trustme
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from urllib3.exceptions import HTTPWarning
from urllib3.util import ALPN_PROTOCOLS, resolve_cert_reqs, resolve_ssl_version
def _has_ipv6(host):
    """Returns True if the system can bind an IPv6 address."""
    sock = None
    has_ipv6 = False
    if socket.has_ipv6:
        try:
            sock = socket.socket(socket.AF_INET6)
            sock.bind((host, 0))
            has_ipv6 = _resolves_to_ipv6('localhost')
        except Exception:
            pass
    if sock:
        sock.close()
    return has_ipv6