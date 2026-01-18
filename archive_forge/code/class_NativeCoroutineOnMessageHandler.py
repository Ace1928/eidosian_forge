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
class NativeCoroutineOnMessageHandler(TestWebSocketHandler):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.sleeping = 0

    async def on_message(self, message):
        if self.sleeping > 0:
            self.write_message('another coroutine is already sleeping')
        self.sleeping += 1
        await gen.sleep(0.01)
        self.sleeping -= 1
        self.write_message(message)