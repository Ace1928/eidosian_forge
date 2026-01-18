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
class StaticDeprecationTests(TestCase):

    def test_addSlashDeprecated(self):
        """
        L{twisted.web.static.addSlash} is deprecated.
        """
        from twisted.web.static import addSlash
        addSlash(DummyRequest([b'']))
        warnings = self.flushWarnings([self.test_addSlashDeprecated])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'twisted.web.static.addSlash was deprecated in Twisted 16.0.0')