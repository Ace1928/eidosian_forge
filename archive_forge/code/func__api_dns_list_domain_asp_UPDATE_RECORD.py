import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_WORLDWIDEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.common.worldwidedns import InvalidDomainName, NonExistentDomain
from libcloud.dns.drivers.worldwidedns import WorldWideDNSError, WorldWideDNSDriver
def _api_dns_list_domain_asp_UPDATE_RECORD(self, method, url, body, headers):
    body = self.fixtures.load('api_dns_list_domain_asp_UPDATE_RECORD')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])