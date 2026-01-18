import shutil
import tempfile
import time
from boto.exception import GSResponseError
from boto.gs.connection import GSConnection
from tests.integration.gs import util
from tests.integration.gs.util import retry
from tests.unit import unittest
def _MakeBucket(self):
    """Creates and returns temporary bucket for testing. After the test, the
        contents of the bucket and the bucket itself will be deleted."""
    b = self._conn.create_bucket(self._MakeBucketName())
    return b