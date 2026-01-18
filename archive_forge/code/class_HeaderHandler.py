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
class HeaderHandler(TestWebSocketHandler):

    def open(self):
        methods_to_test = [functools.partial(self.write, 'This should not work'), functools.partial(self.redirect, 'http://localhost/elsewhere'), functools.partial(self.set_header, 'X-Test', ''), functools.partial(self.set_cookie, 'Chocolate', 'Chip'), functools.partial(self.set_status, 503), self.flush, self.finish]
        for method in methods_to_test:
            try:
                method()
                raise Exception('did not get expected exception')
            except RuntimeError:
                pass
        self.write_message(self.request.headers.get('X-Test', ''))