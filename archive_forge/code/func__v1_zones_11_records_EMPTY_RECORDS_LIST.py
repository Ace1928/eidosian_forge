import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LUADNS
from libcloud.dns.drivers.luadns import LuadnsDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_zones_11_records_EMPTY_RECORDS_LIST(self, method, url, body, headers):
    body = self.fixtures.load('empty_records_list.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])