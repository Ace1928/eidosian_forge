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
def CheckEqual(a, b):
    """Ensures that a and b are equal."""
    self.assertEqual(a, b)
    self.assertFalse(a != b, '%r unexpectedly equals %r' % (a, b))
    self.assertEqual(hash(a), hash(b), 'hash %d of %r unexpectedly not equal to hash %d of %r' % (hash(a), a, hash(b), b))
    self.assertFalse(a < b, '%r unexpectedly less than %r' % (a, b))
    self.assertFalse(b < a, '%r unexpectedly less than %r' % (b, a))
    self.assertLessEqual(a, b)
    self.assertLessEqual(b, a)
    self.assertFalse(a > b, '%r unexpectedly greater than %r' % (a, b))
    self.assertFalse(b > a, '%r unexpectedly greater than %r' % (b, a))
    self.assertGreaterEqual(a, b)
    self.assertGreaterEqual(b, a)