import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _Domain_Info_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('zone_does_not_exist.json')
    return (404, body, {}, httplib.responses[httplib.OK])