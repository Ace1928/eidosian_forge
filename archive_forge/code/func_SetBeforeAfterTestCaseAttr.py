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
def SetBeforeAfterTestCaseAttr():
    TestCase.setUpTestCase = lambda self: None
    TestCase.tearDownTestCase = lambda self: None