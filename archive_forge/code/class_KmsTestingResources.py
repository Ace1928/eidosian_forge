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
class KmsTestingResources(object):
    """Constants for KMS resource names to be used in integration testing."""
    KEYRING_LOCATION = 'us-central1'
    KEYRING_NAME = 'keyring-for-gsutil-integration-tests'
    CONSTANT_KEY_NAME = 'key-for-gsutil-integration-tests'
    CONSTANT_KEY_NAME2 = 'key-for-gsutil-integration-tests2'
    CONSTANT_KEY_NAME_DO_NOT_AUTHORIZE = 'key-for-gsutil-no-auth'
    MUTABLE_KEY_NAME_TEMPLATE = 'cryptokey-for-gsutil-integration-tests-%d%d%d'