from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
class SetUpTearDown(AsyncHTTPTestCase):

    def get_app(self):
        return Application()

    def test(self):
        self.assertTrue(hasattr(self, '_app'))
        self.assertTrue(hasattr(self, 'http_server'))