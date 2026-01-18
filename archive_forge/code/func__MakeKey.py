import shutil
import tempfile
import time
from boto.exception import GSResponseError
from boto.gs.connection import GSConnection
from tests.integration.gs import util
from tests.integration.gs.util import retry
from tests.unit import unittest
def _MakeKey(self, data='', bucket=None, set_contents=True):
    """Creates and returns a Key with provided data. If no bucket is given,
        a temporary bucket is created."""
    if data and (not set_contents):
        raise ValueError('MakeKey called with a non-empty data parameter but set_contents was set to False.')
    if not bucket:
        bucket = self._MakeBucket()
    key_name = self._MakeTempName()
    k = bucket.new_key(key_name)
    if set_contents:
        k.set_contents_from_string(data)
    return k