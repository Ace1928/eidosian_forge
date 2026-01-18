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
def MockCreateObject(self, bucket_name, object_name, contents=''):
    """Creates an object without exercising the API directly."""
    if bucket_name not in self.buckets:
        self.MockCreateBucket(bucket_name)
    self.buckets[bucket_name].CreateObject(object_name, contents=contents)