import shutil
import tempfile
import time
from boto.exception import GSResponseError
from boto.gs.connection import GSConnection
from tests.integration.gs import util
from tests.integration.gs.util import retry
from tests.unit import unittest
def _MakeTempName(self):
    """Creates and returns a temporary name for testing that is likely to be
        unique."""
    return 'boto-gs-test-%s' % repr(time.time()).replace('.', '-')