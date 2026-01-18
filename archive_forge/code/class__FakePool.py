from __future__ import absolute_import, print_function
import os
import shutil
import tempfile
from .Dependencies import cythonize, extended_iglob
from ..Utils import is_package_dir
from ..Compiler import Options
class _FakePool(object):

    def map_async(self, func, args):
        try:
            from itertools import imap
        except ImportError:
            imap = map
        for _ in imap(func, args):
            pass

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass