import datetime
import decimal
import unittest
import warnings
from webtest import TestApp
from wsme import WSRoot, Unset
from wsme import expose, validate
import wsme.types
import wsme.utils
class BodyTypes(object):

    def assertEqual(self, a, b):
        if not a == b:
            raise AssertionError('%s != %s' % (a, b))

    @expose(int, body={wsme.types.text: int})
    @validate(int)
    def setdict(self, body):
        print(body)
        self.assertEqual(type(body), dict)
        self.assertEqual(type(body['test']), int)
        self.assertEqual(body['test'], 10)
        return body['test']

    @expose(int, body=[int])
    @validate(int)
    def setlist(self, body):
        print(body)
        self.assertEqual(type(body), list)
        self.assertEqual(type(body[0]), int)
        self.assertEqual(body[0], 10)
        return body[0]