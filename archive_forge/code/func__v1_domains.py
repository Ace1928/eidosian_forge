import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSIMPLE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.dnsimple import DNSimpleDNSDriver
def _v1_domains(self, method, url, body, headers):
    body = self.fixtures.load('list_domains.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])