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
class ImportObjectTest(unittest.TestCase):

    def test_import_member(self):
        self.assertIs(import_object('tornado.escape.utf8'), utf8)

    def test_import_member_unicode(self):
        self.assertIs(import_object('tornado.escape.utf8'), utf8)

    def test_import_module(self):
        self.assertIs(import_object('tornado.escape'), tornado.escape)

    def test_import_module_unicode(self):
        self.assertIs(import_object('tornado.escape'), tornado.escape)