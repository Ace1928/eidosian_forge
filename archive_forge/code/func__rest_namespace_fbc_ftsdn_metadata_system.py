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
def _rest_namespace_fbc_ftsdn_metadata_system(self, method, url, body, headers):
    if not self.upload_stream_created:
        self.__class__.upload_stream_created = True
        body = self.fixtures.load('not_found.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])
    self.__class__.upload_stream_created = False
    meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
    headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
    return (httplib.OK, '', headers, httplib.responses[httplib.OK])