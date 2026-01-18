from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
class OAuth2AccountClientTest(unittest.TestCase):
    """Unit tests for OAuth2UserAccountClient and OAuth2ServiceAccountClient."""

    def setUp(self):
        self.tempdirs = []
        self.mock_datetime = MockDateTime()
        self.start_time = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
        self.mock_datetime.mock_now = self.start_time

    def testGetAccessTokenUserAccount(self):
        self.client = CreateMockUserAccountClient(self.mock_datetime)
        self._RunGetAccessTokenTest(expected_rapt=RAPT_TOKEN)

    def testGetAccessTokenServiceAccount(self):
        self.client = CreateMockServiceAccountClient(self.mock_datetime)
        self._RunGetAccessTokenTest()

    def _RunGetAccessTokenTest(self, expected_rapt=None):
        """Tests access token gets with self.client."""
        access_token_1 = 'abc123'
        self.assertFalse(self.client.fetched_token)
        token_1 = self.client.GetAccessToken()
        self.assertTrue(self.client.fetched_token)
        self.assertEqual(access_token_1, token_1.token)
        self.assertEqual(self.start_time + datetime.timedelta(minutes=60), token_1.expiry)
        self.assertEqual(token_1.rapt_token, expected_rapt)
        self.client.Reset()
        self.mock_datetime.mock_now = self.start_time + datetime.timedelta(minutes=55)
        token_2 = self.client.GetAccessToken()
        self.assertEqual(token_1, token_2)
        self.assertEqual(access_token_1, token_2.token)
        self.assertFalse(self.client.fetched_token)
        self.client.Reset()
        self.mock_datetime.mock_now = self.start_time + datetime.timedelta(minutes=55, seconds=1)
        self.client.datetime_strategy = self.mock_datetime
        token_3 = self.client.GetAccessToken()
        self.assertTrue(self.client.fetched_token)
        self.assertEqual(self.mock_datetime.mock_now + datetime.timedelta(minutes=60), token_3.expiry)
        self.assertEqual(token_3.rapt_token, expected_rapt)