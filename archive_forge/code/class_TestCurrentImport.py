import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestCurrentImport(LineTestCase):

    def setUp(self):
        self.func = current_import

    def test_simple(self):
        self.assertAccess('import <path|>')
        self.assertAccess('import <p|ath>')
        self.assertAccess('import |path')
        self.assertAccess('import path, <another|>')
        self.assertAccess('import path another|')
        self.assertAccess('if True: import <path|>')
        self.assertAccess('if True: import <xml.dom.minidom|>')
        self.assertAccess('if True: import <xml.do|m.minidom>')
        self.assertAccess('if True: import <xml.do|m.minidom> as something')