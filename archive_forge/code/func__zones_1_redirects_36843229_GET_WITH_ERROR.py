import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_POINTDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.pointdns import PointDNSDriver, PointDNSException
def _zones_1_redirects_36843229_GET_WITH_ERROR(self, method, url, body, headers):
    body = self.fixtures.load('redirect_error.json')
    return (httplib.METHOD_NOT_ALLOWED, body, {}, httplib.responses[httplib.METHOD_NOT_ALLOWED])