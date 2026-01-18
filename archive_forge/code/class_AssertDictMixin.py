import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
class AssertDictMixin:

    def assert_dict_equals(self, expected, actual):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        self.assertEqual(len(expected_keys), len(actual_keys))
        self.assertEqual(0, len(expected_keys - actual_keys))
        for key in expected:
            self.assertEqual(expected[key], actual[key])