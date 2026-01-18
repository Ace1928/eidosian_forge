import sys
import unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI
from libcloud.dns.drivers.gandi import GandiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_gandi import BaseGandiMockHttp
def _xmlrpc__domain_zone_list(self, method, url, body, headers):
    body = self.fixtures.load('list_zones.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])