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
class OpenCoroutineHandler(TestWebSocketHandler):

    def initialize(self, test, **kwargs):
        super().initialize(**kwargs)
        self.test = test
        self.open_finished = False

    @gen.coroutine
    def open(self):
        yield self.test.message_sent.wait()
        yield gen.sleep(0.01)
        self.open_finished = True

    def on_message(self, message):
        if not self.open_finished:
            raise Exception('on_message called before open finished')
        self.write_message('ok')