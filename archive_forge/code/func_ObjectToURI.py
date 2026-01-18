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
def ObjectToURI(obj, *suffixes):
    """Returns the storage URI string for a given StorageUri or file object.

  Args:
    obj: The object to get the URI from. Can be a file object, a subclass of
         boto.storage_uri.StorageURI, or a string. If a string, it is assumed to
         be a local on-disk path.
    *suffixes: Suffixes to append. For example, ObjectToUri(bucketuri, 'foo')
               would return the URI for a key name 'foo' inside the given
               bucket.

  Returns:
    Storage URI string.
  """
    if is_file(obj):
        return 'file://{}'.format(os.path.abspath(os.path.join(obj.name, *suffixes)))
    if isinstance(obj, six.string_types):
        return 'file://{}'.format(os.path.join(obj, *suffixes))
    uri = six.ensure_text(obj.uri)
    if suffixes:
        suffixes_list = [six.ensure_text(suffix) for suffix in suffixes]
        uri = _NormalizeURI('/'.join([uri] + suffixes_list))
    if uri.endswith('/'):
        uri = uri[:-1]
    return uri