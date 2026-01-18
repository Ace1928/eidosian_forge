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
def GetGsutilSubprocess(cmd, env_vars=None):
    """Returns a subprocess.Popen object for for running a gsutil command."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    envstr = dict()
    for k, v in six.iteritems(env):
        envstr[six.ensure_str(k)] = six.ensure_str(v)
    preexec_fn = os.setsid if hasattr(os, 'setsid') else None
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=envstr, preexec_fn=preexec_fn)