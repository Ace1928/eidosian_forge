import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _Record_Info_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('get_record.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])