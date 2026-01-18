from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
class BazsController(RestController):
    data = [[['zero-zero-zero']]]
    final = FinalController()

    @expose()
    def get_one(self, foo_id, bar_id, id):
        return self.data[int(foo_id)][int(bar_id)][int(id)]

    @expose()
    def post(self):
        return 'POST-GRAND-CHILD'

    @expose()
    def put(self, id):
        return 'PUT-GRAND-CHILD'