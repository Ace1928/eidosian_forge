import os
import tempfile
import unittest
from base64 import b64decode
from launchpadlib.testing.helpers import (
from launchpadlib.credentials import (
class TestAccessToken(unittest.TestCase):
    """Tests for the AccessToken class."""

    def test_from_string(self):
        access_token = AccessToken.from_string('oauth_token_secret=secret%3Dpassword&oauth_token=lock%26key')
        self.assertEqual('lock&key', access_token.key)
        self.assertEqual('secret=password', access_token.secret)
        self.assertIsNone(access_token.context)

    def test_from_string_with_context(self):
        access_token = AccessToken.from_string('oauth_token_secret=secret%3Dpassword&oauth_token=lock%26key&lp.context=firefox')
        self.assertEqual('lock&key', access_token.key)
        self.assertEqual('secret=password', access_token.secret)
        self.assertEqual('firefox', access_token.context)