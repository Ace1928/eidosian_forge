import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LUADNS
from libcloud.dns.drivers.luadns import LuadnsDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_zones_13_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('zone_does_not_exist.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])