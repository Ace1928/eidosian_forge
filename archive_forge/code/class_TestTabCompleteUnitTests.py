from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import time
from gslib.command import CreateOrGetGsutilLogger
from gslib.tab_complete import CloudObjectCompleter
from gslib.tab_complete import TAB_COMPLETE_CACHE_TTL
from gslib.tab_complete import TabCompletionCache
import gslib.tests.testcase as testcase
from gslib.tests.util import ARGCOMPLETE_AVAILABLE
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.boto_util import GetTabCompletionCacheFilename
@unittest.skipUnless(ARGCOMPLETE_AVAILABLE, 'Tab completion requires argcomplete')
class TestTabCompleteUnitTests(testcase.unit_testcase.GsUtilUnitTestCase):
    """Unit tests for tab completion."""

    def test_cached_results(self):
        """Tests tab completion results returned from cache."""
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            request = 'gs://prefix'
            cached_results = ['gs://prefix1', 'gs://prefix2']
            _WriteTabCompletionCache(request, cached_results)
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual(cached_results, results)

    def test_expired_cached_results(self):
        """Tests tab completion results not returned from cache when too old."""
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            bucket_base_name = self.MakeTempName('bucket')
            bucket_name = bucket_base_name + '-suffix'
            self.CreateBucket(bucket_name)
            request = '%s://%s' % (self.default_provider, bucket_base_name)
            expected_result = '%s://%s/' % (self.default_provider, bucket_name)
            cached_results = ['//%s1' % bucket_name, '//%s2' % bucket_name]
            _WriteTabCompletionCache(request, cached_results, time.time() - TAB_COMPLETE_CACHE_TTL)
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual([expected_result], results)

    def test_prefix_caching(self):
        """Tests tab completion results returned from cache with prefix match.

    If the tab completion prefix is an extension of the cached prefix, tab
    completion should return results from the cache that start with the prefix.
    """
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            cached_prefix = 'gs://prefix'
            cached_results = ['gs://prefix-first', 'gs://prefix-second']
            _WriteTabCompletionCache(cached_prefix, cached_results)
            request = 'gs://prefix-f'
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual(['gs://prefix-first'], results)

    def test_prefix_caching_boundary(self):
        """Tests tab completion prefix caching not spanning directory boundaries.

    If the tab completion prefix is an extension of the cached prefix, but is
    not within the same bucket/sub-directory then the cached results should not
    be used.
    """
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            object_uri = self.CreateObject(object_name='subdir/subobj', contents=b'test data')
            cached_prefix = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
            cached_results = ['%s://%s/subdir' % (self.default_provider, object_uri.bucket_name)]
            _WriteTabCompletionCache(cached_prefix, cached_results)
            request = '%s://%s/subdir/' % (self.default_provider, object_uri.bucket_name)
            expected_result = '%s://%s/subdir/subobj' % (self.default_provider, object_uri.bucket_name)
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual([expected_result], results)

    def test_prefix_caching_no_results(self):
        """Tests tab completion returning empty result set using cached prefix.

    If the tab completion prefix is an extension of the cached prefix, but does
    not match any of the cached results then no remote request should be made
    and an empty result set should be returned.
    """
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            object_uri = self.CreateObject(object_name='obj', contents=b'test data')
            cached_prefix = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
            cached_results = []
            _WriteTabCompletionCache(cached_prefix, cached_results)
            request = '%s://%s/o' % (self.default_provider, object_uri.bucket_name)
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual([], results)

    def test_prefix_caching_partial_results(self):
        """Tests tab completion prefix matching ignoring partial cached results.

    If the tab completion prefix is an extension of the cached prefix, but the
    cached result set is partial, the cached results should not be used because
    the matching results for the prefix may be incomplete.
    """
        with SetBotoConfigForTest([('GSUtil', 'state_dir', self.CreateTempDir())]):
            object_uri = self.CreateObject(object_name='obj', contents=b'test data')
            cached_prefix = '%s://%s/' % (self.default_provider, object_uri.bucket_name)
            cached_results = []
            _WriteTabCompletionCache(cached_prefix, cached_results, partial_results=True)
            request = '%s://%s/o' % (self.default_provider, object_uri.bucket_name)
            completer = CloudObjectCompleter(self.MakeGsUtilApi())
            results = completer(request)
            self.assertEqual([str(object_uri)], results)