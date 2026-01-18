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
def _v1_MossoCloudFS(self, method, url, body, headers):
    headers = copy.deepcopy(self.base_headers)
    if method == 'GET':
        body = self.fixtures.load('list_containers.json')
        status_code = httplib.OK
    elif method == 'HEAD':
        body = self.fixtures.load('meta_data.json')
        status_code = httplib.NO_CONTENT
        headers.update({'x-account-container-count': '10', 'x-account-object-count': '400', 'x-account-bytes-used': '1234567'})
    elif method == 'POST':
        body = ''
        status_code = httplib.NO_CONTENT
    return (status_code, body, headers, httplib.responses[httplib.OK])