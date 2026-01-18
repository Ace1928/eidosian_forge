import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def _filters_DescribeVpcs(self, method, url, body, headers):
    expected_params = {'Filter.1.Name': 'dhcp-options-id', 'Filter.1.Value.1': 'dopt-7eded312', 'Filter.2.Name': 'cidr', 'Filter.2.Value.1': '192.168.51.0/24'}
    self.assertUrlContainsQueryParams(url, expected_params)
    body = self.fixtures.load('describe_vpcs.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])