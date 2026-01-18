from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import sys
import tempfile
import six
import boto
from boto.utils import get_utf8able_str
from gslib import project_id
from gslib import wildcard_iterator
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command_runner import CommandRunner
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.constants import UTF8
from gslib.utils.text_util import print_to_fd
def CreateBucket(self, bucket_name=None, test_objects=0, storage_class=None, provider='gs'):
    """Creates a test bucket.

    The bucket and all of its contents will be deleted after the test.

    Args:
      bucket_name: Create the bucket with this name. If not provided, a
                   temporary test bucket name is constructed.
      test_objects: The number of objects that should be placed in the bucket or
                    a list of object names to place in the bucket. Defaults to
                    0.
      storage_class: storage class to use. If not provided we us standard.
      provider: string provider to use, default gs.

    Returns:
      StorageUri for the created bucket.
    """
    bucket_name = bucket_name or self.MakeTempName('bucket')
    bucket_uri = boto.storage_uri('%s://%s' % (provider, bucket_name.lower()), suppress_consec_slashes=False, bucket_storage_uri_class=util.GSMockBucketStorageUri)
    bucket_uri.create_bucket(storage_class=storage_class)
    self.bucket_uris.append(bucket_uri)
    try:
        iter(test_objects)
    except TypeError:
        test_objects = [self.MakeTempName('obj') for _ in range(test_objects)]
    for i, name in enumerate(test_objects):
        self.CreateObject(bucket_uri=bucket_uri, object_name=name, contents='test {}'.format(i).encode(UTF8))
    return bucket_uri