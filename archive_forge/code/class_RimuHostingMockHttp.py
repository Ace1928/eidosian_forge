import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.compute import TestCaseMixin
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.rimuhosting import RimuHostingNodeDriver
class RimuHostingMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('rimuhosting')

    def _r_orders(self, method, url, body, headers):
        body = self.fixtures.load('r_orders.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _r_pricing_plans(self, method, url, body, headers):
        body = self.fixtures.load('r_pricing_plans.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _r_distributions(self, method, url, body, headers):
        body = self.fixtures.load('r_distributions.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _r_orders_new_vps(self, method, url, body, headers):
        body = self.fixtures.load('r_orders_new_vps.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _r_orders_order_88833465_api_ivan_net_nz_vps(self, method, url, body, headers):
        body = self.fixtures.load('r_orders_order_88833465_api_ivan_net_nz_vps.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _r_orders_order_88833465_api_ivan_net_nz_vps_running_state(self, method, url, body, headers):
        body = self.fixtures.load('r_orders_order_88833465_api_ivan_net_nz_vps_running_state.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])