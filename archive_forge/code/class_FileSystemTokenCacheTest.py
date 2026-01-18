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
class FileSystemTokenCacheTest(unittest.TestCase):
    """Unit tests for FileSystemTokenCache."""

    def setUp(self):
        self.cache = oauth2_client.FileSystemTokenCache()
        self.start_time = datetime.datetime(2011, 3, 1, 10, 25, 13, 300826)
        self.token_1 = oauth2_client.AccessToken('token1', self.start_time, rapt_token=RAPT_TOKEN)
        self.token_2 = oauth2_client.AccessToken('token2', self.start_time + datetime.timedelta(seconds=492), rapt_token=RAPT_TOKEN)
        self.key = 'token1key'

    def tearDown(self):
        try:
            os.unlink(self.cache.CacheFileName(self.key))
        except:
            pass

    def testPut(self):
        self.cache.PutToken(self.key, self.token_1)
        if not IS_WINDOWS:
            self.assertEqual(384, stat.S_IMODE(os.stat(self.cache.CacheFileName(self.key)).st_mode))

    def testPutGet(self):
        """Tests putting and getting various tokens."""
        self.assertEqual(None, self.cache.GetToken(self.key))
        self.cache.PutToken(self.key, self.token_1)
        cached_token = self.cache.GetToken(self.key)
        self.assertEqual(self.token_1, cached_token)
        self.cache.PutToken(self.key, self.token_2)
        cached_token = self.cache.GetToken(self.key)
        self.assertEqual(self.token_2, cached_token)

    def testGetBadFile(self):
        f = open(self.cache.CacheFileName(self.key), 'w')
        f.write('blah')
        f.close()
        self.assertEqual(None, self.cache.GetToken(self.key))

    def testCacheFileName(self):
        """Tests configuring the cache with a specific file name."""
        cache = oauth2_client.FileSystemTokenCache(path_pattern='/var/run/ccache/token.%(uid)s.%(key)s')
        if IS_WINDOWS:
            uid = '_'
        else:
            uid = os.getuid()
        self.assertEqual('/var/run/ccache/token.%s.abc123' % uid, cache.CacheFileName('abc123'))
        cache = oauth2_client.FileSystemTokenCache(path_pattern='/var/run/ccache/token.%(key)s')
        self.assertEqual('/var/run/ccache/token.abc123', cache.CacheFileName('abc123'))