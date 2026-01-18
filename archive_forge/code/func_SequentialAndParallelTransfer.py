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
def SequentialAndParallelTransfer(func):
    """Decorator for tests that perform file to object transfers, or vice versa.

  This forces the test to run once normally, and again with special boto
  config settings that will ensure that the test follows the parallel composite
  upload and/or sliced object download code paths.

  Args:
    func: Function to wrap.

  Returns:
    Wrapped function.
  """

    @functools.wraps(func)
    def Wrapper(*args, **kwargs):
        func(*args, **kwargs)
        if not RUN_S3_TESTS and UsingCrcmodExtension():
            with SetBotoConfigForTest([('GSUtil', 'parallel_composite_upload_threshold', '1'), ('GSUtil', 'sliced_object_download_threshold', '1'), ('GSUtil', 'sliced_object_download_max_components', '3'), ('GSUtil', 'check_hashes', 'always')]):
                func(*args, **kwargs)
    return Wrapper