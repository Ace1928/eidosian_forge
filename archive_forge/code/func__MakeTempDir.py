import shutil
import tempfile
import time
from boto.exception import GSResponseError
from boto.gs.connection import GSConnection
from tests.integration.gs import util
from tests.integration.gs.util import retry
from tests.unit import unittest
def _MakeTempDir(self):
    """Creates and returns a temporary directory on disk. After the test,
        the contents of the directory and the directory itself will be
        deleted."""
    tmpdir = tempfile.mkdtemp(prefix=self._MakeTempName())
    self._tempdirs.append(tmpdir)
    return tmpdir