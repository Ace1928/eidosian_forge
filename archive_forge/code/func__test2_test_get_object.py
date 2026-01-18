import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
def _test2_test_get_object(self, method, url, body, headers):
    body = self.fixtures.load('list_containers.xml')
    headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-goog-meta-rabbits': 'monkeys', 'content-length': '12345', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])