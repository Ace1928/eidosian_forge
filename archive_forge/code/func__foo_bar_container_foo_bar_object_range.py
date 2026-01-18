import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
def _foo_bar_container_foo_bar_object_range(self, method, url, body, headers):
    body = '0123456789123456789'
    self.assertTrue('x-ms-range' in headers)
    self.assertEqual(headers['x-ms-range'], 'bytes=5-6')
    start_bytes, end_bytes = self._get_start_and_end_bytes_from_range_str(headers['x-ms-range'], body)
    return (httplib.PARTIAL_CONTENT, body[start_bytes:end_bytes + 1], headers, httplib.responses[httplib.PARTIAL_CONTENT])