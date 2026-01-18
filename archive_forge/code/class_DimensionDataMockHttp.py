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
class DimensionDataMockHttp(MockHttp):
    fixtures = LoadBalancerFileFixtures('dimensiondata')

    def _oec_0_9_myaccount_UNAUTHORIZED(self, method, url, body, headers):
        return (httplib.UNAUTHORIZED, '', {}, httplib.responses[httplib.UNAUTHORIZED])

    def _oec_0_9_myaccount(self, method, url, body, headers):
        body = self.fixtures.load('oec_0_9_myaccount.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _oec_0_9_myaccount_INPROGRESS(self, method, url, body, headers):
        body = self.fixtures.load('oec_0_9_myaccount.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_virtualListener(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_virtualListener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_virtualListener_6115469d_a8bb_445b_bb23_d23b5283f2b9(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_virtualListener_6115469d_a8bb_445b_bb23_d23b5283f2b9.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_pool(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_pool.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_pool_4d360b1f_bc2c_4ab7_9884_1f03ba2768f7(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_pool_4d360b1f_bc2c_4ab7_9884_1f03ba2768f7.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_poolMember(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_poolMember.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_poolMember_3dd806a2_c2c8_4c0c_9a4f_5219ea9266c0(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_poolMember_3dd806a2_c2c8_4c0c_9a4f_5219ea9266c0.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_createPool(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_createPool.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_createNode(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_createNode.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_addPoolMember(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_addPoolMember.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_createVirtualListener(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_createVirtualListener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_removePoolMember(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_removePoolMember.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_deleteVirtualListener(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_deleteVirtualListener.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_deletePool(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_deletePool.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_deleteNode(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_deleteNode.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_node(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_node.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_node_34de6ed6_46a4_4dae_a753_2f8d3840c6f9(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_node_34de6ed6_46a4_4dae_a753_2f8d3840c6f9.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_editNode(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_editNode.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_editPool(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_editPool.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_editPoolMember(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_editPoolMember.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_defaultHealthMonitor(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_defaultHealthMonitor.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_defaultPersistenceProfile(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_defaultPersistenceProfile.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _caas_2_3_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_defaultIrule(self, method, url, body, headers):
        body = self.fixtures.load('networkDomainVip_defaultIrule.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])