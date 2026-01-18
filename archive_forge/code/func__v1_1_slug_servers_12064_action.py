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
def _v1_1_slug_servers_12064_action(self, method, url, body, headers):
    if method != 'POST':
        self.fail('HTTP method other than POST to action URL')
    if 'createImage' in json.loads(body):
        return (httplib.ACCEPTED, '', {'location': 'http://127.0.0.1/v1.1/68/images/4949f9ee-2421-4c81-8b49-13119446008b'}, httplib.responses[httplib.ACCEPTED])
    elif 'rescue' in json.loads(body):
        return (httplib.OK, '{"adminPass": "foo"}', {}, httplib.responses[httplib.OK])
    return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])