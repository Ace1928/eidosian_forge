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
def _v1_MossoCloudFS_test_container__7E_test_object(self, method, url, body, headers):
    headers = copy.deepcopy(self.base_headers)
    if method == 'HEAD':
        body = self.fixtures.load('list_container_objects_empty.json')
        status_code = httplib.NO_CONTENT
        headers.update({'content-length': '555', 'last-modified': 'Tue, 25 Jan 2011 22:01:49 GMT', 'etag': '6b21c4a111ac178feacf9ec9d0c71f17', 'x-object-meta-foo-bar': 'test 1', 'x-object-meta-bar-foo': 'test 2', 'content-type': 'application/zip'})
    return (status_code, body, headers, httplib.responses[httplib.OK])