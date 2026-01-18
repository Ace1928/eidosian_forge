from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import functools
import locale
import logging
import os
import pkgutil
import posixpath
import re
import io
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import six
from six.moves import urllib
from six.moves import cStringIO
import boto
import crcmod
import gslib
from gslib.kms_api import KmsApi
from gslib.project_id import PopulateProjectId
import mock_storage_service  # From boto/tests/integration/s3
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadException
from gslib.lazy_wrapper import LazyWrapper
import gslib.tests as gslib_tests
from gslib.utils import posix_util
from gslib.utils.boto_util import UsingCrcmodExtension, HasUserSpecifiedGsHost
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import MakeHumanReadable
@contextmanager
def _SetBotoConfigFileForTest(boto_config_path):
    """Sets a given file as the boto config file for a single test.

  This function applies only the configuration in boto_config_path and will
  ignore existing configuration. It should not be called directly by tests;
  instead, use SetBotoConfigForTest.

  Args:
    boto_config_path: Path to config file to use.

  Yields:
    When configuration has been applied, and again when reverted.
  """
    try:
        old_boto_config_env_variable = os.environ['BOTO_CONFIG']
        boto_config_was_set = True
    except KeyError:
        boto_config_was_set = False
    os.environ['BOTO_CONFIG'] = boto_config_path
    try:
        yield
    finally:
        if boto_config_was_set:
            os.environ['BOTO_CONFIG'] = old_boto_config_env_variable
        else:
            os.environ.pop('BOTO_CONFIG', None)