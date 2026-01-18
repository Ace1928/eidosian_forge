import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
def _test2_get_object_no_content_length(self, method, url, body, headers):
    body = self.fixtures.load('list_containers.xml')
    headers = {'content-type': 'application/zip', 'etag': '"e31208wqsdoj329jd"', 'x-amz-meta-rabbits': 'monkeys', 'last-modified': 'Thu, 13 Sep 2012 07:13:22 GMT'}
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])