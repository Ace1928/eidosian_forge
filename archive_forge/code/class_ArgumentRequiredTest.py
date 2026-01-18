import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class ArgumentRequiredTest(testtools.TestCase):

    def setUp(self):
        super(ArgumentRequiredTest, self).setUp()
        self.param = 'test-param'
        self.arg_req = common.ArgumentRequired(self.param)

    def test___init__(self):
        self.assertEqual(self.param, self.arg_req.param)

    def test___str__(self):
        expected = 'Argument "--%s" required.' % self.param
        self.assertEqual(expected, self.arg_req.__str__())