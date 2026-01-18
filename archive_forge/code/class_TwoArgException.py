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
class TwoArgException(Exception):

    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = (a, b)