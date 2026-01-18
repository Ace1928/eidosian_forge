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
def _v2_1337_v2_images_8af1a54e_a1b2_4df8_b747_4bec97abc799_members_e2151b1fe02d4a8a2d1f5fc331522c0a(self, method, url, body, headers):
    if method == 'PUT':
        body = self.fixtures.load('_images_8af1a54e_a1b2_4df8_b747_4bec97abc799_members.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
    else:
        raise NotImplementedError()