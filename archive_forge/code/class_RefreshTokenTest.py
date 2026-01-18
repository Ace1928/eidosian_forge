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
class RefreshTokenTest(unittest.TestCase):
    """Unit tests for refresh tokens."""

    def setUp(self):
        self.mock_datetime = MockDateTime()
        self.start_time = datetime.datetime(2011, 3, 1, 10, 25, 13, 300826)
        self.mock_datetime.mock_now = self.start_time
        self.client = CreateMockUserAccountClient(self.mock_datetime)

    def testUniqeId(self):
        cred_id = self.client.CacheKey()
        self.assertEqual('0720afed6871f12761fbea3271f451e6ba184bf5', cred_id)

    def testGetAuthorizationHeader(self):
        self.assertEqual('Bearer %s' % ACCESS_TOKEN, self.client.GetAuthorizationHeader())