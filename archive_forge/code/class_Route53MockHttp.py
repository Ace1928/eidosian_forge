import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_ROUTE53
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.route53 import Route53DNSDriver
class Route53MockHttp(MockHttp):
    fixtures = DNSFileFixtures('route53')

    def _2012_02_29_hostedzone_47234(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_02_29_hostedzone(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('create_zone.xml')
            return (httplib.CREATED, body, {}, httplib.responses[httplib.OK])
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_02_29_hostedzone_47234_rrset(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_02_29_hostedzone_47234_rrset_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _2012_02_29_hostedzone_4444_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _2012_02_29_hostedzone_47234_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.xml')
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _2012_02_29_hostedzone_47234_rrset_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('invalid_change_batch.xml')
            return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])
        body = self.fixtures.load('record_does_not_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _2012_02_29_hostedzone_47234_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])