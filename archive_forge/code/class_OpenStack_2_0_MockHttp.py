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
class OpenStack_2_0_MockHttp(OpenStack_1_1_MockHttp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        methods1 = OpenStack_1_1_MockHttp.__dict__
        names1 = [m for m in methods1 if m.find('_v1_1') == 0]
        for name in names1:
            method = methods1[name]
            new_name = name.replace('_v1_1_slug_', '_v2_1337_')
            setattr(self, new_name, method_type(method, self, OpenStack_2_0_MockHttp))

    def _v2_0_tenants_UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])