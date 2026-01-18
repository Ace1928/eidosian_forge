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
def _RevertBotoConfig(revert_list):
    """Reverts boto config modifications made by _SetBotoConfig.

  Args:
    revert_list: List of boto config modifications created by calls to
                 _SetBotoConfig.
  """
    sections_to_remove = []
    for section, name, value in revert_list:
        if value is None:
            if name == TEST_BOTO_REMOVE_SECTION:
                sections_to_remove.append(section)
            else:
                boto.config.remove_option(section, name)
        else:
            boto.config.set(section, name, value)
    for section in sections_to_remove:
        boto.config.remove_section(section)