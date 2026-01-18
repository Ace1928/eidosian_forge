from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.cloud_api import ServiceException
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateObjectNotFoundException
class MockCloudApi(object):
    """Simple mock service for buckets/objects that implements Cloud API.

  Also includes some setup functions for tests.
  """

    def __init__(self, provider='gs'):
        self.buckets = {}
        self.provider = provider
        self.status_queue = DiscardMessagesQueue()

    def MockCreateBucket(self, bucket_name):
        """Creates a simple bucket without exercising the API directly."""
        if bucket_name in self.buckets:
            raise ServiceException('Bucket %s already exists.' % bucket_name, status=409)
        self.buckets[bucket_name] = MockBucket(bucket_name)

    def MockCreateVersionedBucket(self, bucket_name):
        """Creates a simple bucket without exercising the API directly."""
        if bucket_name in self.buckets:
            raise ServiceException('Bucket %s already exists.' % bucket_name, status=409)
        self.buckets[bucket_name] = MockBucket(bucket_name, versioned=True)

    def MockCreateObject(self, bucket_name, object_name, contents=''):
        """Creates an object without exercising the API directly."""
        if bucket_name not in self.buckets:
            self.MockCreateBucket(bucket_name)
        self.buckets[bucket_name].CreateObject(object_name, contents=contents)

    def MockCreateObjectWithMetadata(self, apitools_object, contents=''):
        """Creates an object without exercising the API directly."""
        assert apitools_object.bucket, 'No bucket specified for mock object'
        assert apitools_object.name, 'No object name specified for mock object'
        if apitools_object.bucket not in self.buckets:
            self.MockCreateBucket(apitools_object.bucket)
        return self.buckets[apitools_object.bucket].CreateObjectWithMetadata(apitools_object, contents=contents).root_object

    def GetObjectMetadata(self, bucket_name, object_name, generation=None, provider=None, fields=None):
        """See CloudApi class for function doc strings."""
        if generation:
            generation = long(generation)
        if bucket_name in self.buckets:
            bucket = self.buckets[bucket_name]
            if object_name in bucket.objects and bucket.objects[object_name]:
                if generation:
                    if 'versioned' in bucket.objects[object_name]:
                        for obj in bucket.objects[object_name]['versioned']:
                            if obj.root_object.generation == generation:
                                return obj.root_object
                    if 'live' in bucket.objects[object_name]:
                        if bucket.objects[object_name]['live'].root_object.generation == generation:
                            return bucket.objects[object_name]['live'].root_object
                elif 'live' in bucket.objects[object_name]:
                    return bucket.objects[object_name]['live'].root_object
            raise CreateObjectNotFoundException(404, self.provider, bucket_name, object_name)
        raise CreateBucketNotFoundException(404, self.provider, bucket_name)