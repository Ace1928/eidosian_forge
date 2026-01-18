import copy
import boto
import base64
import re
import six
from hashlib import md5
from boto.utils import compute_md5
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import write_to_fd
from boto.s3.prefix import Prefix
from boto.compat import six
class MockConnection(object):

    def __init__(self, aws_access_key_id=NOT_IMPL, aws_secret_access_key=NOT_IMPL, is_secure=NOT_IMPL, port=NOT_IMPL, proxy=NOT_IMPL, proxy_port=NOT_IMPL, proxy_user=NOT_IMPL, proxy_pass=NOT_IMPL, host=NOT_IMPL, debug=NOT_IMPL, https_connection_factory=NOT_IMPL, calling_format=NOT_IMPL, path=NOT_IMPL, provider='s3', bucket_class=NOT_IMPL):
        self.buckets = {}
        self.provider = MockProvider(provider)

    def create_bucket(self, bucket_name, headers=NOT_IMPL, location=NOT_IMPL, policy=NOT_IMPL, storage_class=NOT_IMPL):
        if bucket_name in self.buckets:
            raise boto.exception.StorageCreateError(409, 'BucketAlreadyOwnedByYou', '<Message>Your previous request to create the named bucket succeeded and you already own it.</Message>')
        mock_bucket = MockBucket(name=bucket_name, connection=self)
        self.buckets[bucket_name] = mock_bucket
        return mock_bucket

    def delete_bucket(self, bucket, headers=NOT_IMPL):
        if bucket not in self.buckets:
            raise boto.exception.StorageResponseError(404, 'NoSuchBucket', '<Message>no such bucket</Message>')
        del self.buckets[bucket]

    def get_bucket(self, bucket_name, validate=NOT_IMPL, headers=NOT_IMPL):
        if bucket_name not in self.buckets:
            raise boto.exception.StorageResponseError(404, 'NoSuchBucket', 'Not Found')
        return self.buckets[bucket_name]

    def get_all_buckets(self, headers=NOT_IMPL):
        return six.itervalues(self.buckets)