import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.dns.drivers.vultr import VultrDNSDriver, VultrDNSDriverV1
from libcloud.test.file_fixtures import DNSFileFixtures
def _v1_dns_list_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
    body = self.fixtures.load('list_domains.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])