import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
def _2012_02_29_hostedzone_47234_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
    body = self.fixtures.load('zone_does_not_exist.xml')
    return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])