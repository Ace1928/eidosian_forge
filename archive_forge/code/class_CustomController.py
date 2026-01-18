from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
class CustomController(RestController):
    _custom_actions = {'detail': ['GET'], 'create': ['POST'], 'update': ['PUT'], 'remove': ['DELETE']}

    @expose()
    def detail(self):
        return 'DETAIL'

    @expose()
    def create(self):
        return 'CREATE'

    @expose()
    def update(self, id):
        return id

    @expose()
    def remove(self, id):
        return id