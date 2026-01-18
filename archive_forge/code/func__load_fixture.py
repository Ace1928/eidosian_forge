import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import Provider
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.providers import get_driver
from libcloud.loadbalancer.drivers.cloudstack import CloudStackLBDriver
def _load_fixture(self, fixture):
    body = self.fixtures.load(fixture)
    return (body, json.loads(body))