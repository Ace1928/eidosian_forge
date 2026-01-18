import os
import pkgutil
import shutil
import tempfile
import httplib2
class Httplib2(object):

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp()

    def __enter__(self):
        _monkey_patch_httplib2(self._tmpdir)
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        shutil.rmtree(self._tmpdir)