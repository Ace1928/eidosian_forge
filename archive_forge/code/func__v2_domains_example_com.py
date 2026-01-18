import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.vultr import VultrDNSDriver, VultrDNSDriverV2
from libcloud.test.file_fixtures import DNSFileFixtures
def _v2_domains_example_com(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    if method == 'DELETE':
        return (httplib.NO_CONTENT, body, {}, httplib.responses[httplib.NO_CONTENT])