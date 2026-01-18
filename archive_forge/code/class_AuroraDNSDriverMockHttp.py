import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.dns.base import Zone
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError
from libcloud.test.secrets import DNS_PARAMS_AURORADNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.auroradns import AuroraDNSDriver, AuroraDNSHealthCheckType
class AuroraDNSDriverMockHttp(MockHttp):
    fixtures = DNSFileFixtures('auroradns')

    def _zones(self, method, url, body, headers):
        if method == 'POST':
            body_json = json.loads(body)
            if body_json['name'] == 'exists.example.com':
                return (httplib.CONFLICT, body, {}, httplib.responses[httplib.CONFLICT])
            body = self.fixtures.load('zone_example_com.json')
        else:
            body = self.fixtures.load('zone_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_HTTP_FORBIDDEN(self, method, url, body, headers):
        body = '{}'
        return (httplib.FORBIDDEN, body, {}, httplib.responses[httplib.FORBIDDEN])

    def _zones_example_com(self, method, url, body, headers):
        body = None
        if method == 'GET':
            body = self.fixtures.load('zone_example_com.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_nonexists_example_com(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_ffb62570_8414_4578_a346_526b44e320b7(self, method, url, body, headers):
        body = self.fixtures.load('zone_example_com.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_ffb62570_8414_4578_a346_526b44e320b7_records(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zone_example_com_record_localhost.json')
        else:
            body = self.fixtures.load('zone_example_com_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_ffb62570_8414_4578_a346_526b44e320b7_health_checks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zone_example_com_health_check.json')
        else:
            body = self.fixtures.load('zone_example_com_health_checks.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _zones_1(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_records(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_1_records_1(self, method, url, body, headers):
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.NOT_FOUND])

    def _zones_ffb62570_8414_4578_a346_526b44e320b7_records_5592f1ff(self, method, url, body, headers):
        body = None
        if method == 'GET':
            body = self.fixtures.load('zone_example_com_record_localhost.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])