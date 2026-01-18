import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def _regions(self, method, url, body, headers):
    if 'pageToken' in url or 'filter' in url:
        body = self.fixtures.load('regions-paged-2.json')
    elif 'maxResults' in url:
        body = self.fixtures.load('regions-paged-1.json')
    else:
        body = self.fixtures.load('regions.json')
    return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])