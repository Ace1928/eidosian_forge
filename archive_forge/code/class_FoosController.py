from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
class FoosController(RestController):

    @expose()
    def _lookup(self, id_, *remainder):
        return (FooController(), remainder)

    @expose()
    def get_all(self):
        return 'FOOS'

    @expose()
    def post(self):
        return 'POST_FOOS'