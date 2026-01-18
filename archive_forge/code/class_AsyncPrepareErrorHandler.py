import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
class AsyncPrepareErrorHandler(RequestHandler):

    @gen.coroutine
    def prepare(self):
        yield gen.moment
        raise HTTPError(403)

    def get(self):
        self.finish('ok')