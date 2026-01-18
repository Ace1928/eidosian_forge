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
def MockCreateObjectWithMetadata(self, apitools_object, contents=''):
    """Creates an object without exercising the API directly."""
    assert apitools_object.bucket, 'No bucket specified for mock object'
    assert apitools_object.name, 'No object name specified for mock object'
    if apitools_object.bucket not in self.buckets:
        self.MockCreateBucket(apitools_object.bucket)
    return self.buckets[apitools_object.bucket].CreateObjectWithMetadata(apitools_object, contents=contents).root_object