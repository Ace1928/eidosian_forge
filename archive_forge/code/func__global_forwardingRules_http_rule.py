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
def _global_forwardingRules_http_rule(self, method, url, body, headers):
    if method == 'DELETE':
        body = self.fixtures.load('global_forwardingRules_http_rule_delete.json')
    else:
        body = self.fixtures.load('global_forwardingRules_http_rule.json')
    return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])