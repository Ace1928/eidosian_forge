from contextlib import contextmanager
import os
import shutil
import socket
import stat
import tempfile
import unittest
import warnings
from lazr.restfulclient.resource import ServiceRoot
from launchpadlib.credentials import (
from launchpadlib import uris
import launchpadlib.launchpad
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import UnencryptedFileCredentialStore
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestLaunchpadLoginWithCredentialsFileFromEnvVariable(unittest.TestCase):

    def test_filename(self):
        ignore, filename = tempfile.mkstemp()
        os.environ['LP_CREDENTIALS_FILE'] = filename
        launchpad = NoNetworkLaunchpad.login_with(application_name='not important')
        self.assertIsInstance(launchpad.credential_store, UnencryptedFileCredentialStore)
        self.assertEqual(launchpad.credential_store.filename, filename)
        os.unsetenv('LP_CREDENTIALS_FILE')
        del os.environ['LP_CREDENTIALS_FILE']
        self.assertIsNone(os.environ.get('LP_CREDENTIALS_FILE'))
        os.remove(filename)