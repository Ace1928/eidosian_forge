import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
def _GET_RECORD_RECORD_DOES_NOT_EXIST_domain_list(self, method, url, body, headers):
    body = self.fixtures.load('get_zone.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])