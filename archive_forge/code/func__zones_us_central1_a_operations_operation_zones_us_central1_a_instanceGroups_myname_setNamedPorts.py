import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_setNamedPorts(self, method, url, body, headers):
    """Redirects from _zones_us_central1_a_instanceGroups_myname_setNamedPorts"""
    body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_myname_setNamedPorts.json')
    return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])