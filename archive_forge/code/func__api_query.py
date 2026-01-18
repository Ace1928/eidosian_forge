import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def _api_query(self, method, url, body, headers):
    assert method == 'GET'
    if 'type=user' in url:
        self.assertTrue('page=2' in url)
        self.assertTrue('filter=(name==jrambo)' in url)
        self.assertTrue('sortDesc=startDate')
        body = self.fixtures.load('api_query_user.xml')
    elif 'type=group' in url:
        body = self.fixtures.load('api_query_group.xml')
    elif 'type=vm' in url and 'filter=(name==testVm2)' in url:
        body = self.fixtures.load('api_query_vm.xml')
    else:
        raise AssertionError('Unexpected query type')
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])