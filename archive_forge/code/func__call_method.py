from __future__ import print_function
import collections
import contextlib
import gzip
import json
import logging
import sys
import time
import zlib
from datetime import datetime, timedelta
from io import BytesIO
from tornado import httputil
from tornado.web import RequestHandler
from urllib3.packages.six import binary_type, ensure_str
from urllib3.packages.six.moves.http_client import responses
from urllib3.packages.six.moves.urllib.parse import urlsplit
def _call_method(self):
    """Call the correct method in this class based on the incoming URI"""
    req = self.request
    req.params = {}
    for k, v in req.arguments.items():
        req.params[k] = next(iter(v))
    path = req.path[:]
    if not path.startswith('/'):
        path = urlsplit(path).path
    target = path[1:].split('/', 1)[0]
    method = getattr(self, target, self.index)
    resp = method(req)
    if dict(resp.headers).get('Connection') == 'close':
        pass
    resp(self)