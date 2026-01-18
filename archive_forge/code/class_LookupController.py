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
class LookupController(object):

    def __init__(self, someID):
        self.someID = someID

    @expose()
    def index(self, req, resp):
        return self.someID