import os
import stat
import time
from ... import osutils
from ...errors import BzrError
from ...tests import TestCaseInTempDir
from ...tests.features import OsFifoFeature
from ..hashcache import HashCache
def _fingerprint(self, abspath, fs=None):
    entry = self._files[abspath]
    return (len(entry[0]), entry[1], entry[1], 10, 20, stat.S_IFREG | 384)