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
class HaltOneComponentCopyCallbackHandler(object):
    """Test callback handler for stopping part of a sliced download."""

    def __init__(self, halt_at_byte):
        self._last_progress_byte = None
        self._halt_at_byte = halt_at_byte

    def call(self, current_progress_byte, total_size_unused):
        """Forcibly exits if the passed the halting point since the last call."""
        if self._last_progress_byte is not None and self._last_progress_byte < self._halt_at_byte < current_progress_byte:
            sys.stderr.write('Halting transfer.\r\n')
            raise ResumableDownloadException('Artifically halting download.')
        self._last_progress_byte = current_progress_byte