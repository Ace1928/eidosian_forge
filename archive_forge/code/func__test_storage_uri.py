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
@staticmethod
def _test_storage_uri(uri_str, default_scheme='file', debug=0, validate=True):
    """Convenience method for instantiating a testing instance of StorageUri.

    This makes it unnecessary to specify
    bucket_storage_uri_class=mock_storage_service.MockBucketStorageUri.
    Also naming the factory method this way makes it clearer in the test
    code that StorageUri needs to be set up for testing.

    Args, Returns, and Raises are same as for boto.storage_uri(), except there's
    no bucket_storage_uri_class arg.

    Args:
      uri_str: Uri string to create StorageUri for.
      default_scheme: Default scheme for the StorageUri
      debug: debug level to pass to the underlying connection (0..3)
      validate: If True, validate the resource that the StorageUri refers to.

    Returns:
      StorageUri based on the arguments.
    """
    return boto.storage_uri(uri_str, default_scheme, debug, validate, util.GSMockBucketStorageUri)