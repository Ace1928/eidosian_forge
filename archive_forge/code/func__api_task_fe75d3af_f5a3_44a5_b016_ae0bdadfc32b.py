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
def _api_task_fe75d3af_f5a3_44a5_b016_ae0bdadfc32b(self, method, url, body, headers):
    body = self.fixtures.load('api_task_fe75d3af_f5a3_44a5_b016_ae0bdadfc32b.xml')
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])