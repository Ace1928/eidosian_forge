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
def _v1_MossoCloudFS_foo_bar_container_foo_test_stream_data_seek(self, method, url, body, headers):
    hasher = hashlib.md5()
    hasher.update(b'123456789')
    hash_value = hasher.hexdigest()
    headers = {}
    headers.update(self.base_headers)
    headers['etag'] = hash_value
    body = 'test'
    return (httplib.CREATED, body, headers, httplib.responses[httplib.OK])