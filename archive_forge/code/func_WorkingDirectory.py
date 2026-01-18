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
def WorkingDirectory(new_working_directory):
    """Changes the working directory for the duration of a 'with' call.

  Args:
    new_working_directory: The directory to switch to before executing wrapped
      code. A None value indicates that no switching is necessary.

  Yields:
    Once after working directory has been changed.
  """
    prev_working_directory = None
    try:
        prev_working_directory = os.getcwd()
    except OSError:
        pass
    if new_working_directory:
        os.chdir(new_working_directory)
    try:
        yield
    finally:
        if new_working_directory and prev_working_directory:
            os.chdir(prev_working_directory)