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
class GsutilApiUnitTestClassMapFactory(object):
    """Class map factory for use in unit tests.

  BotoTranslation is used for all cases so that GSMockBucketStorageUri can
  be used to communicate with the mock XML service.
  """

    @classmethod
    def GetClassMap(cls):
        """Returns a class map for use in unit tests."""
        gs_class_map = {ApiSelector.XML: BotoTranslation, ApiSelector.JSON: BotoTranslation}
        s3_class_map = {ApiSelector.XML: BotoTranslation}
        class_map = {'gs': gs_class_map, 's3': s3_class_map}
        return class_map