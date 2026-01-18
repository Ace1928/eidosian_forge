from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
import six
import tempfile
from gslib import wildcard_iterator
from gslib.exception import InvalidUrlError
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetDummyProjectForUnitTest
class CloudWildcardIteratorTests(testcase.GsUtilUnitTestCase):
    """Unit tests for CloudWildcardIterator."""

    def setUp(self):
        """Creates 2 mock buckets, each containing 4 objects, including 1 nested."""
        super(CloudWildcardIteratorTests, self).setUp()
        self.immed_child_obj_names = ['abcd', 'abdd', 'ade$']
        self.all_obj_names = ['abcd', 'abdd', 'ade$', 'nested1/nested2/xyz1', 'nested1/nested2/xyz2', 'nested1/nested2xyz1', 'nested1/nfile_abc']
        self.base_bucket_uri = self.CreateBucket()
        self.prefix_bucket_name = '%s_' % self.base_bucket_uri.bucket_name[:61]
        self.base_uri_str = suri(self.base_bucket_uri)
        self.base_uri_str = self.base_uri_str.replace(self.base_bucket_uri.bucket_name, self.prefix_bucket_name)
        self.test_bucket0_uri = self.CreateBucket(bucket_name='%s0' % self.prefix_bucket_name)
        self.test_bucket0_obj_uri_strs = set()
        for obj_name in self.all_obj_names:
            obj_uri = self.CreateObject(bucket_uri=self.test_bucket0_uri, object_name=obj_name, contents='')
            self.test_bucket0_obj_uri_strs.add(suri(obj_uri))
        self.test_bucket1_uri = self.CreateBucket(bucket_name='%s1' % self.prefix_bucket_name)
        self.test_bucket1_obj_uri_strs = set()
        for obj_name in self.all_obj_names:
            obj_uri = self.CreateObject(bucket_uri=self.test_bucket1_uri, object_name=obj_name, contents='')
            self.test_bucket1_obj_uri_strs.add(suri(obj_uri))
        self.test_bucket2_uri = self.CreateBucket(bucket_name='%s2' % self.prefix_bucket_name)
        self.test_bucket2_obj_uri_strs = set()
        object_list = ['f.txt', 'double/f.txt', 'double/zf.txt', 'double/foo/f.txt', 'double/foo/zf.txt', 'double/bar/f.txt', 'double/bar/zf.txt']
        for obj_name in object_list:
            obj_uri = self.CreateObject(bucket_uri=self.test_bucket2_uri, object_name=obj_name, contents='')
            self.test_bucket2_obj_uri_strs.add(suri(obj_uri))

    def testNoOpObjectIterator(self):
        """Tests that bucket-only URI iterates just that one URI."""
        results = list(self._test_wildcard_iterator(self.test_bucket0_uri).IterBuckets(bucket_fields=['id']))
        self.assertEqual(1, len(results))
        self.assertEqual(str(self.test_bucket0_uri), str(results[0]))

    def testMatchingAllObjects(self):
        """Tests matching all objects, based on wildcard."""
        actual_obj_uri_strs = set((six.ensure_text(str(u)) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(self.test_bucket0_obj_uri_strs, actual_obj_uri_strs)

    def testMatchingObjectSubset(self):
        """Tests matching a subset of objects, based on wildcard."""
        exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('abcd')), str(self.test_bucket0_uri.clone_replace_name('abdd'))])
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('ab??')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testMatchingNonWildcardedUri(self):
        """Tests matching a single named object."""
        exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('abcd'))])
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('abcd')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testWildcardedObjectUriWithVsWithoutPrefix(self):
        """Tests that wildcarding w/ and w/o server prefix get same result."""
        with_prefix_uri_strs = set((str(u) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('abcd')).IterAll(expand_top_level_buckets=True)))
        no_prefix_uri_strs = set((str(u) for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('?bcd')).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(with_prefix_uri_strs, no_prefix_uri_strs)

    def testWildcardedObjectUriNestedSubdirMatch(self):
        """Tests wildcarding with a nested subdir."""
        uri_strs = set()
        prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('*')):
            if blr.IsPrefix():
                prefixes.add(blr.root_object)
            else:
                uri_strs.add(blr.url_string)
        exp_obj_uri_strs = set([suri(self.test_bucket0_uri, x) for x in self.immed_child_obj_names])
        self.assertEqual(exp_obj_uri_strs, uri_strs)
        self.assertEqual(1, len(prefixes))
        self.assertTrue('nested1/' in prefixes)

    def testWildcardPlusSubdirMatch(self):
        """Tests gs://bucket/*/subdir matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('*/nested1')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set()
        expected_prefixes = set(['nested1/'])
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)

    def testWildcardPlusSubdirSubdirMatch(self):
        """Tests gs://bucket/*/subdir/* matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('*/nested2/*')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set([self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz1').uri, self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz2').uri])
        expected_prefixes = set()
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)

    def testNoMatchingWildcardedObjectUri(self):
        """Tests that get back an empty iterator for non-matching wildcarded URI."""
        res = list(self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('*x0')).IterAll(expand_top_level_buckets=True))
        self.assertEqual(0, len(res))

    def testWildcardedInvalidObjectUri(self):
        """Tests that we raise an exception for wildcarded invalid URI."""
        try:
            for unused_ in self._test_wildcard_iterator('badscheme://asdf').IterAll(expand_top_level_buckets=True):
                self.assertFalse('Expected InvalidUrlError not raised.')
        except InvalidUrlError as e:
            self.assertTrue(e.message.find('Unrecognized scheme') != -1)

    def testSingleMatchWildcardedBucketUri(self):
        """Tests matching a single bucket based on a wildcarded bucket URI."""
        exp_obj_uri_strs = set([suri(self.test_bucket1_uri) + self.test_bucket1_uri.delim])
        with SetDummyProjectForUnitTest():
            actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s*1' % self.base_uri_str).IterBuckets(bucket_fields=['id'])))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testMultiMatchWildcardedBucketUri(self):
        """Tests matching a multiple buckets based on a wildcarded bucket URI."""
        exp_obj_uri_strs = set([suri(self.test_bucket0_uri) + self.test_bucket0_uri.delim, suri(self.test_bucket1_uri) + self.test_bucket1_uri.delim, suri(self.test_bucket2_uri) + self.test_bucket2_uri.delim])
        with SetDummyProjectForUnitTest():
            actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s*' % self.base_uri_str).IterBuckets(bucket_fields=['id'])))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testWildcardBucketAndObjectUri(self):
        """Tests matching with both bucket and object wildcards."""
        exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('abcd'))])
        with SetDummyProjectForUnitTest():
            actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s0*/abc*' % self.base_uri_str).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testWildcardUpToFinalCharSubdirPlusObjectName(self):
        """Tests wildcard subd*r/obj name."""
        exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz1'))])
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%snested1/nest*2/xyz1' % self.test_bucket0_uri.uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testPostRecursiveWildcard(self):
        """Tests wildcard containing ** followed by an additional wildcard."""
        exp_obj_uri_strs = set([str(self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz2'))])
        actual_obj_uri_strs = set((str(u) for u in self._test_wildcard_iterator('%s**/*y*2' % self.test_bucket0_uri.uri).IterAll(expand_top_level_buckets=True)))
        self.assertEqual(exp_obj_uri_strs, actual_obj_uri_strs)

    def testWildcardFields(self):
        """Tests that wildcard w/fields specification returns correct fields."""
        blrs = set((u for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(bucket_listing_fields=['timeCreated'])))
        self.assertTrue(len(blrs))
        for blr in blrs:
            self.assertTrue(blr.root_object and blr.root_object.timeCreated)
        blrs = set((u for u in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**')).IterAll(bucket_listing_fields=['generation'])))
        self.assertTrue(len(blrs))
        for blr in blrs:
            self.assertTrue(blr.root_object and (not blr.root_object.timeCreated))

    def testDoesNotStripDelimiterForDoubleWildcard(self):
        """Tests gs://bucket/*/subdir matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket0_uri.clone_replace_name('**/xyz*')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set([self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz1').uri, self.test_bucket0_uri.clone_replace_name('nested1/nested2/xyz2').uri])
        expected_prefixes = set()
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)

    def testDoubleWildcardAfterBucket(self):
        """Tests gs://bucket/**/object matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket2_uri.clone_replace_name('**/f.txt')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set([self.test_bucket2_uri.clone_replace_name('f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/foo/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/bar/f.txt').uri])
        expected_prefixes = set()
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)

    def testDoubleWildcardAfterPrefix(self):
        """Tests gs://bucket/dir/**/object matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket2_uri.clone_replace_name('double/**/f.txt')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set([self.test_bucket2_uri.clone_replace_name('double/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/foo/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/bar/f.txt').uri])
        expected_prefixes = set()
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)

    def testDoubleWildcardBeforeAndAfterPrefix(self):
        """Tests gs://bucket/**/dir/**/object matching."""
        actual_uri_strs = set()
        actual_prefixes = set()
        for blr in self._test_wildcard_iterator(self.test_bucket2_uri.clone_replace_name('**/double/**/f.txt')):
            if blr.IsPrefix():
                actual_prefixes.add(blr.root_object)
            else:
                actual_uri_strs.add(blr.url_string)
        expected_uri_strs = set([self.test_bucket2_uri.clone_replace_name('double/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/foo/f.txt').uri, self.test_bucket2_uri.clone_replace_name('double/bar/f.txt').uri])
        expected_prefixes = set()
        self.assertEqual(expected_prefixes, actual_prefixes)
        self.assertEqual(expected_uri_strs, actual_uri_strs)