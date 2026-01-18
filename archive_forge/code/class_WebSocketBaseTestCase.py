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
class WebSocketBaseTestCase(AsyncHTTPTestCase):

    def setUp(self):
        super().setUp()
        self.conns_to_close = []

    def tearDown(self):
        for conn in self.conns_to_close:
            conn.close()
        super().tearDown()

    @gen.coroutine
    def ws_connect(self, path, **kwargs):
        ws = (yield websocket_connect('ws://127.0.0.1:%d%s' % (self.get_http_port(), path), **kwargs))
        self.conns_to_close.append(ws)
        raise gen.Return(ws)