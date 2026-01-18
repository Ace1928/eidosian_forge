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