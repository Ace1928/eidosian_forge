import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
class ErrorHook(PecanHook):

    def on_error(self, state, e):
        run_hook.append('error')
        r = webob.Response()
        r.text = 'on_error'
        return r