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
def _getAssertEqualityFunc(self, first, second):
    try:
        return super(TestCase, self)._getAssertEqualityFunc(first, second)
    except AttributeError:
        test_method = getattr(self, '_testMethodName', 'assertTrue')
        super(TestCase, self).__init__(test_method)
    return super(TestCase, self)._getAssertEqualityFunc(first, second)