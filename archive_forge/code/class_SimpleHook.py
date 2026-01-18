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
class SimpleHook(PecanHook):

    def __init__(self, id):
        self.id = str(id)

    def on_route(self, state):
        run_hook.append('on_route' + self.id)

    def before(self, state):
        run_hook.append('before' + self.id)

    def after(self, state):
        run_hook.append('after' + self.id)

    def on_error(self, state, e):
        run_hook.append('error' + self.id)