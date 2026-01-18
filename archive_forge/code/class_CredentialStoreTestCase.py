import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class CredentialStoreTestCase(unittest.TestCase):

    def make_credential(self, consumer_key):
        """Helper method to make a fake credential."""
        return Credentials('app name', consumer_secret='consumer_secret:42', access_token=AccessToken(consumer_key, 'access_secret:168'))