import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LUADNS
from libcloud.dns.drivers.luadns import LuadnsDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_zones_31_records_31_GET_RECORD_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('get_record.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])