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
def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a(self, method, url, body, headers):
    status = httplib.OK
    if method == 'GET':
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a.xml')
        status = httplib.OK
    elif method == 'DELETE':
        body = self.fixtures.load('api_task_b034df55_fe81_4798_bc81_1f0fd0ead450.xml')
        status = httplib.ACCEPTED
    return (status, body, headers, httplib.responses[status])