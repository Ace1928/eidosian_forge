import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
def _assertLogged(self, expected):
    """
        Asserts that a given log message occurred with an expected message.
        """
    logItem = self.catcher.pop()
    self.assertEqual(logItem['message'][0], expected)
    self.assertEqual(self.catcher, [], f'An additional log occurred: {logItem!r}')