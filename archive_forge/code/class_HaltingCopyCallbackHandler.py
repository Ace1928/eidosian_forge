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
class HaltingCopyCallbackHandler(object):
    """Test callback handler for intentionally stopping a resumable transfer."""

    def __init__(self, is_upload, halt_at_byte):
        self._is_upload = is_upload
        self._halt_at_byte = halt_at_byte

    def call(self, total_bytes_transferred, total_size):
        """Forcibly exits if the transfer has passed the halting point.

    Note that this function is only called when the conditions in
    gslib.progress_callback.ProgressCallbackWithTimeout.Progress are met, so
    self._halt_at_byte is only precise if it's divisible by
    gslib.progress_callback._START_BYTES_PER_CALLBACK.
    """
        if total_bytes_transferred >= self._halt_at_byte:
            sys.stderr.write('Halting transfer after byte %s. %s/%s transferred.\r\n' % (self._halt_at_byte, MakeHumanReadable(total_bytes_transferred), MakeHumanReadable(total_size)))
            if self._is_upload:
                raise ResumableUploadException('Artifically halting upload.')
            else:
                raise ResumableDownloadException('Artifically halting download.')