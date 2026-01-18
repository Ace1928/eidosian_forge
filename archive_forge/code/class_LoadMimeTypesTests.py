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
class LoadMimeTypesTests(TestCase):
    """
    Tests for the MIME type loading routine.

    @cvar UNSET: A sentinel to signify that C{self.paths} has not been set by
        the mock init.
    """
    UNSET = object()

    def setUp(self):
        self.paths = self.UNSET

    def _fakeInit(self, paths):
        """
        A mock L{mimetypes.init} that records the value of the passed C{paths}
        argument.

        @param paths: The paths that will be recorded.
        """
        self.paths = paths

    def test_defaultArgumentIsNone(self):
        """
        By default, L{None} is passed to C{mimetypes.init}.
        """
        static.loadMimeTypes(init=self._fakeInit)
        self.assertIdentical(self.paths, None)

    def test_extraLocationsWork(self):
        """
        Passed MIME type files are passed to C{mimetypes.init}.
        """
        paths = ['x', 'y', 'z']
        static.loadMimeTypes(paths, init=self._fakeInit)
        self.assertIdentical(self.paths, paths)

    def test_usesGlobalInitFunction(self):
        """
        By default, C{mimetypes.init} is called.
        """
        if getattr(inspect, 'signature', None):
            signature = inspect.signature(static.loadMimeTypes)
            self.assertIs(signature.parameters['init'].default, mimetypes.init)
        else:
            args, _, _, defaults = inspect.getargspec(static.loadMimeTypes)
            defaultInit = defaults[args.index('init')]
            self.assertIs(defaultInit, mimetypes.init)