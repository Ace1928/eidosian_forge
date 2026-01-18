import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ZONOMI
from libcloud.dns.drivers.zonomi import ZonomiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _app_dns_dyndns_jsp_LIST_RECORDS_SUCCESS(self, method, url, body, headers):
    body = self.fixtures.load('list_records.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])