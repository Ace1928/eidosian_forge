import asyncio
import contextlib
import functools
import socket
import traceback
import typing
import unittest
from tornado.concurrent import Future
from tornado import gen
from tornado.httpclient import HTTPError, HTTPRequest
from tornado.locks import Event
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, gen_test, bind_unused_port, ExpectLog
from tornado.web import Application, RequestHandler
from tornado.websocket import (
class HeaderEchoHandler(TestWebSocketHandler):

    def set_default_headers(self):
        self.set_header('X-Extra-Response-Header', 'Extra-Response-Value')

    def prepare(self):
        for k, v in self.request.headers.get_all():
            if k.lower().startswith('x-test'):
                self.set_header(k, v)