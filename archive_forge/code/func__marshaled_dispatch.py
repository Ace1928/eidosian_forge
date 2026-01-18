from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
def _marshaled_dispatch(self, data, dispatch_method=None, path=None):
    try:
        response = self.dispatchers[path]._marshaled_dispatch(data, dispatch_method, path)
    except BaseException as exc:
        response = dumps(Fault(1, '%s:%s' % (type(exc), exc)), encoding=self.encoding, allow_none=self.allow_none)
        response = response.encode(self.encoding, 'xmlcharrefreplace')
    return response