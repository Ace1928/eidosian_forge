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
class SubprotocolHandler(TestWebSocketHandler):

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.select_subprotocol_called = False

    def select_subprotocol(self, subprotocols):
        if self.select_subprotocol_called:
            raise Exception('select_subprotocol called twice')
        self.select_subprotocol_called = True
        if 'goodproto' in subprotocols:
            return 'goodproto'
        return None

    def open(self):
        if not self.select_subprotocol_called:
            raise Exception('select_subprotocol not called')
        self.write_message('subprotocol=%s' % self.selected_subprotocol)