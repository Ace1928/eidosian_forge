import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
class BeforeAfterTestCaseMeta(type):
    """Adds setUpTestCase() and tearDownTestCase() methods.

  These may be needed for setup and teardown of shared fixtures usually because
  such fixtures are expensive to setup and teardown (eg Perforce clients).  When
  using such fixtures, care should be taken to keep each test as independent as
  possible (eg via the use of sandboxes).

  Example:

    class MyTestCase(basetest.TestCase):

      __metaclass__ = basetest.BeforeAfterTestCaseMeta

      @classmethod
      def setUpTestCase(cls):
        cls._resource = foo.ReallyExpensiveResource()

      @classmethod
      def tearDownTestCase(cls):
        cls._resource.Destroy()

      def testSomething(self):
        self._resource.Something()
        ...
  """
    _test_loader = unittest.defaultTestLoader

    def __init__(cls, name, bases, dict):
        super(BeforeAfterTestCaseMeta, cls).__init__(name, bases, dict)
        test_names = set(cls._test_loader.getTestCaseNames(cls))
        cls.__tests_to_run = None
        BeforeAfterTestCaseMeta.SetSetUpAttr(cls, test_names)
        BeforeAfterTestCaseMeta.SetTearDownAttr(cls)
        BeforeAfterTestCaseMeta.SetTestMethodAttrs(cls, test_names)
        BeforeAfterTestCaseMeta.SetBeforeAfterTestCaseAttr()

    @staticmethod
    def SetMethod(cls, method_name, replacement):
        """Like setattr, but also preserves name, doc, and module metadata."""
        original = getattr(cls, method_name)
        replacement.__name__ = original.__name__
        replacement.__doc__ = original.__doc__
        replacement.__module__ = original.__module__
        setattr(cls, method_name, replacement)

    @staticmethod
    def SetSetUpAttr(cls, test_names):
        """Wraps setUp() with per-class setUp() functionality."""
        cls_setUp = cls.setUp

        def setUp(self):
            """Function that will encapsulate and replace cls.setUp()."""
            leaf = self.__class__
            if leaf.__tests_to_run is None:
                leaf.__tests_to_run = set(test_names)
                self.setUpTestCase()
            cls_setUp(self)
        BeforeAfterTestCaseMeta.SetMethod(cls, 'setUp', setUp)

    @staticmethod
    def SetTearDownAttr(cls):
        """Wraps tearDown() with per-class tearDown() functionality."""
        cls_tearDown = cls.tearDown

        def tearDown(self):
            """Function that will encapsulate and replace cls.tearDown()."""
            cls_tearDown(self)
            leaf = self.__class__
            if leaf.__tests_to_run is not None and (not leaf.__tests_to_run) and (leaf == cls):
                leaf.__tests_to_run = None
                self.tearDownTestCase()
        BeforeAfterTestCaseMeta.SetMethod(cls, 'tearDown', tearDown)

    @staticmethod
    def SetTestMethodAttrs(cls, test_names):
        """Makes each test method first remove itself from the remaining set."""
        for test_name in test_names:
            cls_test = getattr(cls, test_name)

            def test(self, cls_test=cls_test, test_name=test_name):
                leaf = self.__class__
                leaf.__tests_to_run.discard(test_name)
                return cls_test(self)
            BeforeAfterTestCaseMeta.SetMethod(cls, test_name, test)

    @staticmethod
    def SetBeforeAfterTestCaseAttr():
        TestCase.setUpTestCase = lambda self: None
        TestCase.tearDownTestCase = lambda self: None