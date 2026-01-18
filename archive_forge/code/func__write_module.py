import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def _write_module(self, name, encoding, contents):
    """Create Python module on disk with contents in given encoding"""
    try:
        codecs.lookup(encoding)
    except LookupError:
        self.skipTest('Encoding unsupported by implementation: %r' % encoding)
    f = codecs.open(os.path.join(self.dir, name + '.py'), 'w', encoding)
    try:
        f.write(contents)
    finally:
        f.close()