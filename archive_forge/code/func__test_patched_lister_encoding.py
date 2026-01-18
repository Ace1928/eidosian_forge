from mock import patch, Mock
import unittest
from boto.s3.bucket import ResultSet
from boto.s3.bucketlistresultset import multipart_upload_lister
from boto.s3.bucketlistresultset import versioned_bucket_lister
def _test_patched_lister_encoding(self, inner_method, outer_method):
    bucket = Mock()
    call_args = []
    first = ResultSet()
    first.append('foo')
    first.next_key_marker = 'a+b'
    first.is_truncated = True
    second = ResultSet()
    second.append('bar')
    second.is_truncated = False
    pages = [first, second]

    def return_pages(**kwargs):
        call_args.append(kwargs)
        return pages.pop(0)
    setattr(bucket, inner_method, return_pages)
    results = list(outer_method(bucket, encoding_type='url'))
    self.assertEqual(['foo', 'bar'], results)
    self.assertEqual('a b', call_args[1]['key_marker'])