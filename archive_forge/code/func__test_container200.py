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
def _test_container200(self, method, url, body, headers):
    body = ''
    if method != 'HEAD':
        return (httplib.BAD_REQUEST, body, self.base_headers, httplib.responses[httplib.BAD_REQUEST])
    headers = {}
    headers['etag'] = '0x8CFB877BB56A6FB'
    headers['last-modified'] = 'Fri, 04 Jan 2013 09:48:06 GMT'
    headers['x-ms-lease-status'] = 'unlocked'
    headers['x-ms-lease-state'] = 'available'
    headers['x-ms-meta-meta1'] = 'value1'
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])