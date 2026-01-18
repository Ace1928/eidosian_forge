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
class TestConfig3B(TestConfig3):

    def initialize(self, b=None):
        self.b = b