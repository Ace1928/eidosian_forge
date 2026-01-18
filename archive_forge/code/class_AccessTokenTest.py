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
class AccessTokenTest(unittest.TestCase):
    """Unit tests for access token functions."""

    def testShouldRefresh(self):
        """Tests that token.ShouldRefresh returns the correct value."""
        mock_datetime = MockDateTime()
        start = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
        expiry = start + datetime.timedelta(minutes=60)
        token = oauth2_client.AccessToken('foo', expiry, datetime_strategy=mock_datetime)
        mock_datetime.mock_now = start
        self.assertFalse(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=54)
        self.assertFalse(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=55)
        self.assertFalse(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=55, seconds=1)
        self.assertTrue(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=61)
        self.assertTrue(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=58)
        self.assertFalse(token.ShouldRefresh(time_delta=120))
        mock_datetime.mock_now = start + datetime.timedelta(minutes=58, seconds=1)
        self.assertTrue(token.ShouldRefresh(time_delta=120))

    def testShouldRefreshNoExpiry(self):
        """Tests token.ShouldRefresh with no expiry time."""
        mock_datetime = MockDateTime()
        start = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
        token = oauth2_client.AccessToken('foo', None, datetime_strategy=mock_datetime)
        mock_datetime.mock_now = start
        self.assertFalse(token.ShouldRefresh())
        mock_datetime.mock_now = start + datetime.timedelta(minutes=472)
        self.assertFalse(token.ShouldRefresh())

    def testSerialization(self):
        """Tests token serialization."""
        expiry = datetime.datetime(2011, 3, 1, 11, 25, 13, 300826)
        token = oauth2_client.AccessToken('foo', expiry, rapt_token=RAPT_TOKEN)
        serialized_token = token.Serialize()
        LOG.debug('testSerialization: serialized_token=%s', serialized_token)
        token2 = oauth2_client.AccessToken.UnSerialize(serialized_token)
        self.assertEqual(token, token2)