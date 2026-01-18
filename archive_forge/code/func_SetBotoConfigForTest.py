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
def SetBotoConfigForTest(boto_config_list, use_existing_config=True):
    """Sets the input list of boto configs for the duration of a 'with' clause.

  This preserves any existing boto configuration unless it is overwritten in
  the provided boto_config_list.

  Args:
    boto_config_list: list of tuples of:
        (boto config section to set, boto config name to set, value to set)
    use_existing_config: If True, apply boto_config_list to the existing
        configuration, preserving any original values unless they are
        overwritten. Otherwise, apply boto_config_list to a blank configuration.

  Yields:
    Once after config is set.
  """
    revert_configs = []
    tmp_filename = None
    try:
        tmp_fd, tmp_filename = tempfile.mkstemp(prefix='gsutil-temp-cfg')
        os.close(tmp_fd)
        if use_existing_config:
            for boto_config in boto_config_list:
                boto_value = boto_config[2]
                if six.PY3:
                    if isinstance(boto_value, bytes):
                        boto_value = boto_value.decode(UTF8)
                _SetBotoConfig(boto_config[0], boto_config[1], boto_value, revert_configs)
            with open(tmp_filename, 'w') as tmp_file:
                boto.config.write(tmp_file)
        else:
            _WriteSectionDictToFile(_SectionDictFromConfigList(boto_config_list), tmp_filename)
        with _SetBotoConfigFileForTest(tmp_filename):
            yield
    finally:
        _RevertBotoConfig(revert_configs)
        if tmp_filename:
            try:
                os.remove(tmp_filename)
            except OSError:
                pass