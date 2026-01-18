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
def _test_container(self, method, url, body, headers):
    query_string = urlparse.urlsplit(url).query
    query = parse_qs(query_string)
    if 'marker' not in query:
        body = self.fixtures.load('list_objects_1.xml')
    else:
        body = self.fixtures.load('list_objects_2.xml')
    return (httplib.OK, body, self.base_headers, httplib.responses[httplib.OK])