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
def _api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection.xml')
        status = httplib.OK
    else:
        body = self.fixtures.load('put_api_vApp_vm_dd75d1d3_5b7b_48f0_aff3_69622ab7e045_guestCustomizationSection.xml')
        status = httplib.ACCEPTED
    return (status, body, headers, httplib.responses[status])