import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.loadbalancer.drivers.dimensiondata import DimensionDataLBDriver as DimensionData
def _caas_2_4_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_pool_4d360b1f_bc2c_4ab7_9884_1f03ba2768f7(self, method, url, body, headers):
    body = self.fixtures.load('networkDomainVip_pool_4d360b1f_bc2c_4ab7_9884_1f03ba2768f7.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])