from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
class ArgReplacerTest(unittest.TestCase):

    def setUp(self):

        def function(x, y, callback=None, z=None):
            pass
        self.replacer = ArgReplacer(function, 'callback')

    def test_omitted(self):
        args = (1, 2)
        kwargs = dict()
        self.assertIs(self.replacer.get_old_value(args, kwargs), None)
        self.assertEqual(self.replacer.replace('new', args, kwargs), (None, (1, 2), dict(callback='new')))

    def test_position(self):
        args = (1, 2, 'old', 3)
        kwargs = dict()
        self.assertEqual(self.replacer.get_old_value(args, kwargs), 'old')
        self.assertEqual(self.replacer.replace('new', args, kwargs), ('old', [1, 2, 'new', 3], dict()))

    def test_keyword(self):
        args = (1,)
        kwargs = dict(y=2, callback='old', z=3)
        self.assertEqual(self.replacer.get_old_value(args, kwargs), 'old')
        self.assertEqual(self.replacer.replace('new', args, kwargs), ('old', (1,), dict(y=2, callback='new', z=3)))