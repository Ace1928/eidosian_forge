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
class KeyringTest(unittest.TestCase):
    """Base class for tests that use the keyring."""

    def setUp(self):
        assert_keyring_not_imported()
        launchpadlib.credentials.keyring = InMemoryKeyring()

    def tearDown(self):
        del launchpadlib.credentials.keyring