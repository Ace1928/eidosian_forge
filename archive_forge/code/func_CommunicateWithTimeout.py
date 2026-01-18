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
def CommunicateWithTimeout(process, stdin=None):
    if stdin is not None:
        if six.PY3:
            if not isinstance(stdin, bytes):
                stdin = stdin.encode(UTF8)
        else:
            stdin = stdin.encode(UTF8)
    comm_kwargs = {'input': stdin}

    def Kill():
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    if six.PY3:
        comm_kwargs['timeout'] = 180
    else:
        timer = threading.Timer(180, Kill)
        timer.start()
    c_out = process.communicate(**comm_kwargs)
    if not six.PY3:
        timer.cancel()
    try:
        c_out = [six.ensure_text(output) for output in c_out]
    except UnicodeDecodeError:
        c_out = [six.ensure_text(output, locale.getpreferredencoding(False)) for output in c_out]
    return c_out