import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
def _test_get_object(self, method, url, body, headers):
    self.base_headers.update({'accept-ranges': 'bytes', 'connection': 'keep-alive', 'content-length': '0', 'content-type': 'application/octet-stream', 'date': 'Sat, 16 Jan 2016 15:38:14 GMT', 'etag': '"D41D8CD98F00B204E9800998ECF8427E"', 'last-modified': 'Fri, 15 Jan 2016 14:43:15 GMT', 'server': 'AliyunOSS', 'x-oss-object-type': 'Normal', 'x-oss-request-id': '569A63E6257784731E3D877F', 'x-oss-meta-rabbits': 'monkeys'})
    return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])