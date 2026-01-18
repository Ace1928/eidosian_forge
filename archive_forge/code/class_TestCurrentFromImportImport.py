import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentFromImportImport(LineTestCase):

    def setUp(self):
        self.func = current_from_import_import

    def test_simple(self):
        self.assertAccess('from sys import <path|>')
        self.assertAccess('from sys import <p|ath>')
        self.assertAccess('from sys import |path')
        self.assertAccess('from sys| import path')
        self.assertAccess('from s|ys import path')
        self.assertAccess('from |sys import path')
        self.assertAccess('from xml.dom import <N|ode>')
        self.assertAccess('from xml.dom import Node.as|d')