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
def _ClearHoldsOnObjectAndWaitForRetentionDuration(self, bucket_uri, object_name):
    """Removes Holds on test objects and waits till retention duration is over.

    This method makes sure that object is not under active Temporary Hold or
    Release Hold. It also waits (up to 1 minute) till retention duration for the
    object is over. This is necessary for cleanup, otherwise such test objects
    cannot be deleted.

    It's worth noting that tests should do their best to remove holds and wait
    for objects' retention period on their own and this is just a fallback.
    Additionally, Tests should not use retention duration longer than 1 minute,
    preferably only few seconds in order to avoid lengthening test execution
    time unnecessarily.

    Args:
      bucket_uri: bucket's uri.
      object_name: object's name.
    """
    object_metadata = self.json_api.GetObjectMetadata(bucket_uri.bucket_name, object_name, fields=['timeCreated', 'temporaryHold', 'eventBasedHold'])
    object_uri = '{}{}'.format(bucket_uri, object_name)
    if object_metadata.temporaryHold:
        self.RunGsUtil(['retention', 'temp', 'release', object_uri])
    if object_metadata.eventBasedHold:
        self.RunGsUtil(['retention', 'event', 'release', object_uri])
    retention_policy = self.json_api.GetBucket(bucket_uri.bucket_name, fields=['retentionPolicy']).retentionPolicy
    retention_period = retention_policy.retentionPeriod if retention_policy is not None else 0
    if retention_period <= 60:
        time.sleep(retention_period)
    else:
        raise CommandException('Retention duration is too large for bucket "{}". Use shorter durations for Retention duration in tests'.format(bucket_uri))