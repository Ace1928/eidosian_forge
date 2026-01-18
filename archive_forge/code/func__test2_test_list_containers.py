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
def _test2_test_list_containers(self, method, url, body, headers):
    body = self.fixtures.load('list_containers.xml')
    headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])