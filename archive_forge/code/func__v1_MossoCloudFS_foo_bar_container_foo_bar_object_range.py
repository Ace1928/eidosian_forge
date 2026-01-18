import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
def _v1_MossoCloudFS_foo_bar_container_foo_bar_object_range(self, method, url, body, headers):
    if method == 'GET':
        body = '0123456789123456789'
        self.assertTrue('Range' in headers)
        self.assertEqual(headers['Range'], 'bytes=5-6')
        start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['Range'], body)
        return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], self.base_headers, httplib.responses[httplib.PARTIAL_CONTENT])