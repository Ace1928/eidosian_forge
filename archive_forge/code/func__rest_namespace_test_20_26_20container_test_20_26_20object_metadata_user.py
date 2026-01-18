import sys
import base64
import os.path
import unittest
import libcloud.utils.files
from libcloud.test import MockHttp, make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.atmos import AtmosDriver, AtmosConnection
from libcloud.storage.drivers.dummy import DummyIterator
def _rest_namespace_test_20_26_20container_test_20_26_20object_metadata_user(self, method, url, body, headers):
    meta = {'md5': '6b21c4a111ac178feacf9ec9d0c71f17', 'foo-bar': 'test 1', 'bar-foo': 'test 2'}
    headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
    return (httplib.OK, '', headers, httplib.responses[httplib.OK])