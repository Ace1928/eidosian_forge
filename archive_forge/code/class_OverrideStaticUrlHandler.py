from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
class OverrideStaticUrlHandler(RequestHandler):

    def get(self, path):
        do_include = bool(self.get_argument('include_host'))
        self.include_host = not do_include
        regular_url = self.static_url(path)
        override_url = self.static_url(path, include_host=do_include)
        if override_url == regular_url:
            return self.write(str(False))
        protocol = self.request.protocol + '://'
        protocol_length = len(protocol)
        check_regular = regular_url.find(protocol, 0, protocol_length)
        check_override = override_url.find(protocol, 0, protocol_length)
        if do_include:
            result = check_override == 0 and check_regular == -1
        else:
            result = check_override == -1 and check_regular == 0
        self.write(str(result))