import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GODADDY
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.godaddy import GoDaddyDNSDriver
def _v1_domains_aperture_platform_com_records(self, method, url, body, headers):
    body = self.fixtures.load('v1_domains_aperture_platform_com_records.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])