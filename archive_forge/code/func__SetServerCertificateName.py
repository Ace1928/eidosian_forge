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
def _SetServerCertificateName(self, method, url, body, headers):
    params = {'RegionId': self.test.region, 'ServerCertificateId': self.test.cert_id, 'ServerCertificateName': self.test.cert_name}
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('set_server_certificate_name.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])