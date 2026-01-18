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
class TestRequestTokenAuthorizationEngine(unittest.TestCase):
    """Tests for the RequestTokenAuthorizationEngine class."""

    def test_app_must_be_identified(self):
        self.assertRaises(ValueError, NoNetworkAuthorizationEngine, SERVICE_ROOT)

    def test_application_name_identifies_app(self):
        NoNetworkAuthorizationEngine(SERVICE_ROOT, application_name='name')

    def test_consumer_name_identifies_app(self):
        NoNetworkAuthorizationEngine(SERVICE_ROOT, consumer_name='name')

    def test_conflicting_app_identification(self):
        self.assertRaises(ValueError, NoNetworkAuthorizationEngine, SERVICE_ROOT, application_name='name1', consumer_name='name2')
        self.assertRaises(ValueError, NoNetworkAuthorizationEngine, SERVICE_ROOT, application_name='name', consumer_name='name')