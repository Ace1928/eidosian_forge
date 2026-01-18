from tornado.concurrent import Future
from tornado import gen
from tornado.escape import (
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import (
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing  # noqa: F401
import unittest
import urllib.parse
class GetCurrentUserTest(WebTestCase):

    def get_app_kwargs(self):

        class WithoutUserModule(UIModule):

            def render(self):
                return ''

        class WithUserModule(UIModule):

            def render(self):
                return str(self.current_user)
        loader = DictLoader({'without_user.html': '', 'with_user.html': '{{ current_user }}', 'without_user_module.html': '{% module WithoutUserModule() %}', 'with_user_module.html': '{% module WithUserModule() %}'})
        return dict(template_loader=loader, ui_modules={'WithUserModule': WithUserModule, 'WithoutUserModule': WithoutUserModule})

    def tearDown(self):
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):

        class CurrentUserHandler(RequestHandler):

            def prepare(self):
                self.has_loaded_current_user = False

            def get_current_user(self):
                self.has_loaded_current_user = True
                return ''

        class WithoutUserHandler(CurrentUserHandler):

            def get(self):
                self.render_string('without_user.html')
                self.finish(str(self.has_loaded_current_user))

        class WithUserHandler(CurrentUserHandler):

            def get(self):
                self.render_string('with_user.html')
                self.finish(str(self.has_loaded_current_user))

        class CurrentUserModuleHandler(CurrentUserHandler):

            def get_template_namespace(self):
                return self.ui

        class WithoutUserModuleHandler(CurrentUserModuleHandler):

            def get(self):
                self.render_string('without_user_module.html')
                self.finish(str(self.has_loaded_current_user))

        class WithUserModuleHandler(CurrentUserModuleHandler):

            def get(self):
                self.render_string('with_user_module.html')
                self.finish(str(self.has_loaded_current_user))
        return [('/without_user', WithoutUserHandler), ('/with_user', WithUserHandler), ('/without_user_module', WithoutUserModuleHandler), ('/with_user_module', WithUserModuleHandler)]

    @unittest.skip('needs fix')
    def test_get_current_user_is_lazy(self):
        response = self.fetch('/without_user')
        self.assertEqual(response.body, b'False')

    def test_get_current_user_works(self):
        response = self.fetch('/with_user')
        self.assertEqual(response.body, b'True')

    def test_get_current_user_from_ui_module_is_lazy(self):
        response = self.fetch('/without_user_module')
        self.assertEqual(response.body, b'False')

    def test_get_current_user_from_ui_module_works(self):
        response = self.fetch('/with_user_module')
        self.assertEqual(response.body, b'True')