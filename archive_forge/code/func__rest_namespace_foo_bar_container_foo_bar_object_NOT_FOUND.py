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
def _rest_namespace_foo_bar_container_foo_bar_object_NOT_FOUND(self, method, url, body, headers):
    body = self.fixtures.load('not_found.xml')
    return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])