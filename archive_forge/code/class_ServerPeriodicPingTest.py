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
class ServerPeriodicPingTest(WebSocketBaseTestCase):

    def get_app(self):

        class PingHandler(TestWebSocketHandler):

            def on_pong(self, data):
                self.write_message('got pong')
        return Application([('/', PingHandler)], websocket_ping_interval=0.01)

    @gen_test
    def test_server_ping(self):
        ws = (yield self.ws_connect('/'))
        for i in range(3):
            response = (yield ws.read_message())
            self.assertEqual(response, 'got pong')