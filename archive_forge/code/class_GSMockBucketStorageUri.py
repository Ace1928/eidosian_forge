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
class GSMockBucketStorageUri(mock_storage_service.MockBucketStorageUri):

    def connect(self, access_key_id=None, secret_access_key=None):
        return mock_connection

    def compose(self, components, headers=None):
        """Dummy implementation to allow parallel uploads with tests."""
        return self.new_key()

    def get_location(self, headers=None):
        return 'US'

    def get_cors(self, headers=None):
        return boto.gs.cors.Cors()

    def get_encryption_config(self, headers=None):
        return boto.gs.encryptionconfig.EncryptionConfig()

    def get_lifecycle_config(self, headers=None):
        return None

    def get_website_config(self, headers=None):
        return None

    def get_versioning_config(self, headers=None):
        return None