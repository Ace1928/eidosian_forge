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
def _v2_1337_v2_images_f24a3c1b_d52a_4116_91da_25b3eee8f55e(self, method, url, body, headers):
    if method == 'GET' or method == 'PATCH':
        body = self.fixtures.load('_images_f24a3c1b-d52a-4116-91da-25b3eee8f55e.json')
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
    else:
        raise NotImplementedError()