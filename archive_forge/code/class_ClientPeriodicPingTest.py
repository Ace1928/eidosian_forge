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
class ClientPeriodicPingTest(WebSocketBaseTestCase):

    def get_app(self):

        class PingHandler(TestWebSocketHandler):

            def on_ping(self, data):
                self.write_message('got ping')
        return Application([('/', PingHandler)])

    @gen_test
    def test_client_ping(self):
        ws = (yield self.ws_connect('/', ping_interval=0.01))
        for i in range(3):
            response = (yield ws.read_message())
            self.assertEqual(response, 'got ping')
        ws.close()