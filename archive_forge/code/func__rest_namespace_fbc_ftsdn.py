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
def _rest_namespace_fbc_ftsdn(self, method, url, body, headers):
    if self._upload_object_via_stream_first_request:
        self.assertTrue('Range' not in headers)
        self.assertEqual(method, 'POST')
        self._upload_object_via_stream_first_request = False
    else:
        self.assertTrue('Range' in headers)
        self.assertEqual(method, 'PUT')
    return (httplib.OK, '', {}, httplib.responses[httplib.OK])