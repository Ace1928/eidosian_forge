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
def SetTestMethodAttrs(cls, test_names):
    """Makes each test method first remove itself from the remaining set."""
    for test_name in test_names:
        cls_test = getattr(cls, test_name)

        def test(self, cls_test=cls_test, test_name=test_name):
            leaf = self.__class__
            leaf.__tests_to_run.discard(test_name)
            return cls_test(self)
        BeforeAfterTestCaseMeta.SetMethod(cls, test_name, test)