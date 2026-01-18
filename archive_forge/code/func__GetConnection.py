import shutil
import tempfile
import time
from boto.exception import GSResponseError
from boto.gs.connection import GSConnection
from tests.integration.gs import util
from tests.integration.gs.util import retry
from tests.unit import unittest
def _GetConnection(self):
    """Returns the GSConnection object used to connect to GCS."""
    return self._conn