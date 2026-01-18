from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import locale
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import boto
from boto import config
from boto.exception import StorageResponseError
from boto.s3.deletemarker import DeleteMarker
from boto.storage_uri import BucketStorageUri
import gslib
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.kms_api import KmsApi
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.tests.util import USING_JSON_API
import gslib.third_party.storage_apitools.storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
import six
from six.moves import range
def CreateBucketWithRetentionPolicy(self, retention_period_in_seconds, is_locked=None, bucket_name=None):
    """Creates a test bucket with Retention Policy.

    The bucket and all of its contents will be deleted after the test.

    Args:
      retention_period_in_seconds: Retention duration in seconds
      is_locked: Indicates whether Retention Policy should be locked
                 on the bucket or not.
      bucket_name: Create the bucket with this name. If not provided, a
                   temporary test bucket name is constructed.

    Returns:
      StorageUri for the created bucket.
    """
    retention_policy = apitools_messages.Bucket.RetentionPolicyValue(retentionPeriod=retention_period_in_seconds)
    bucket_uri = self.CreateBucket(bucket_name=bucket_name, retention_policy=retention_policy, prefer_json_api=True)
    if is_locked:
        self.RunGsUtil(['retention', 'lock', suri(bucket_uri)], stdin='y')
    self.VerifyRetentionPolicy(bucket_uri, expected_retention_period_in_seconds=retention_period_in_seconds, expected_is_locked=is_locked)
    return bucket_uri