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
def _rest_namespace_test_20_26_20container__metadata_system(self, method, url, body, headers):
    headers = {'x-emc-meta': 'objectid=b21cb59a2ba339d1afdd4810010b0a5aba2ab6b9'}
    return (httplib.OK, '', headers, httplib.responses[httplib.OK])