from __future__ import annotations
import os
import stat
from typing import cast
from unittest import skipIf
from twisted.internet import endpoints, reactor
from twisted.internet.interfaces import IReactorCore, IReactorUNIX
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.threadpool import ThreadPool
from twisted.python.usage import UsageError
from twisted.spread.pb import PBServerFactory
from twisted.trial.unittest import TestCase
from twisted.web import demo
from twisted.web.distrib import ResourcePublisher, UserDirectory
from twisted.web.script import PythonScript
from twisted.web.server import Site
from twisted.web.static import Data, File
from twisted.web.tap import (
from twisted.web.test.requesthelper import DummyRequest
from twisted.web.twcgi import CGIScript
from twisted.web.wsgi import WSGIResource
def _pathOption(self) -> tuple[FilePath[str], File]:
    """
        Helper for the I{--path} tests which creates a directory and creates
        an L{Options} object which uses that directory as its static
        filesystem root.

        @return: A two-tuple of a L{FilePath} referring to the directory and
            the value associated with the C{'root'} key in the L{Options}
            instance after parsing a I{--path} option.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    options = Options()
    options.parseOptions(['--path', path.path])
    root = options['root']
    return (path, root)