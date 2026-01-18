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
def _foo_bar_container_foo_test_upload_lease(self, method, url, body, headers):
    self._assert_content_length_header_is_string(headers=headers)
    action = headers['x-ms-lease-action']
    rheaders = {'x-ms-lease-id': 'someleaseid'}
    body = ''
    if action == 'acquire':
        return (httplib.CREATED, body, rheaders, httplib.responses[httplib.CREATED])
    else:
        if headers.get('x-ms-lease-id', None) != 'someleaseid':
            return (httplib.BAD_REQUEST, body, rheaders, httplib.responses[httplib.BAD_REQUEST])
        return (httplib.OK, body, headers, httplib.responses[httplib.CREATED])