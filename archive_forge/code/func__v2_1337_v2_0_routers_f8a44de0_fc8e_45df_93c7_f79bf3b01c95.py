import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
def _v2_1337_v2_0_routers_f8a44de0_fc8e_45df_93c7_f79bf3b01c95(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('_v2_0__router.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
    if method == 'DELETE':
        body = ''
        return (httplib.NO_CONTENT, body, self.json_content_headers, httplib.responses[httplib.OK])