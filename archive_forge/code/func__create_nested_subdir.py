from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
def _create_nested_subdir(self):
    """Creates a nested subdirectory for use by tests in this module."""
    bucket_uri = self.CreateBucket()
    obj_uris = [self.CreateObject(bucket_uri=bucket_uri, object_name='sub1材/five', contents=b'5five'), self.CreateObject(bucket_uri=bucket_uri, object_name='sub1材/four', contents=b'four'), self.CreateObject(bucket_uri=bucket_uri, object_name='sub1材/sub2/five', contents=b'5five'), self.CreateObject(bucket_uri=bucket_uri, object_name='sub1材/sub2/four', contents=b'four')]
    self.AssertNObjectsInBucket(bucket_uri, 4)
    return (bucket_uri, obj_uris)