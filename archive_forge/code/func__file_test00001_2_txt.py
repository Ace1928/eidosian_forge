import os
import sys
import json
import tempfile
from unittest import mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import b, httplib
from libcloud.utils.files import exhaust_iterator
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.backblaze_b2 import BackblazeB2StorageDriver
def _file_test00001_2_txt(self, method, url, body, headers):
    if method == 'GET':
        body = 'ab'
    else:
        raise AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])