import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.nfsn import NFSNDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _dns_example_com_removeRR_NOT_FOUND(self, method, url, body, headers):
    body = self.fixtures.load('record_not_removed.json')
    return (httplib.NOT_FOUND, body, self.base_headers, httplib.responses[httplib.NOT_FOUND])