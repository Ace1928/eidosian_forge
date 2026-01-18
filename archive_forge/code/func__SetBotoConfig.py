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
def _SetBotoConfig(section, name, value, revert_list):
    """Sets boto configuration temporarily for testing.

  SetBotoConfigForTest should be called by tests instead of this function.
  This will ensure that the configuration is reverted to its original setting
  using _RevertBotoConfig.

  Args:
    section: Boto config section to set
    name: Boto config name to set
    value: Value to set
    revert_list: List for tracking configs to revert.
  """
    prev_value = boto.config.get(section, name, None)
    if not boto.config.has_section(section):
        revert_list.append((section, TEST_BOTO_REMOVE_SECTION, None))
        boto.config.add_section(section)
    revert_list.append((section, name, prev_value))
    if value is None:
        boto.config.remove_option(section, name)
    else:
        boto.config.set(section, name, value)