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
def _rest_namespace_fbc_ftsd_metadata_system(self, method, url, body, headers):
    meta = {'objectid': '322dce3763aadc41acc55ef47867b8d74e45c31d6643', 'size': '555', 'mtime': '2011-01-25T22:01:49Z'}
    headers = {'x-emc-meta': ', '.join([k + '=' + v for k, v in list(meta.items())])}
    return (httplib.OK, '', headers, httplib.responses[httplib.OK])