import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.test.secrets import DNS_PARAMS_ZERIGO
from libcloud.dns.drivers.zerigo import ZerigoError, ZerigoDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class ZerigoMockHttp(MockHttp):
    fixtures = DNSFileFixtures('zerigo')

    def _api_1_1_zones_xml_INVALID_CREDS(self, method, url, body, headers):
        body = 'HTTP Basic: Access denied.\n'
        return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_xml(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {'x-query-count': '1'}, httplib.responses[httplib.OK])

    def _api_1_1_zones_xml_NO_RESULTS(self, method, url, body, headers):
        body = self.fixtures.load('list_zones_no_results.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_hosts_xml(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {'x-query-count': '1'}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_hosts_xml_NO_RESULTS(self, method, url, body, headers):
        body = self.fixtures.load('list_records_no_results.xml')
        return (httplib.OK, body, {'x-query-count': '0'}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_hosts_xml_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_xml(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_4444_xml_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_hosts_23456789_xml(self, method, url, body, headers):
        body = self.fixtures.load('get_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_444_xml_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_xml_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_hosts_28536_xml_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_xml_CREATE_ZONE(self, method, url, body, headers):
        body = self.fixtures.load('create_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_xml_CREATE_ZONE_VALIDATION_ERROR(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_validation_error.xml')
        return (httplib.UNPROCESSABLE_ENTITY, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_hosts_xml_CREATE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('create_record.xml')
        return (httplib.CREATED, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_zones_12345678_xml_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])

    def _api_1_1_hosts_23456789_xml_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.NOT_FOUND, body, {}, httplib.responses[httplib.OK])
    "\n    def (self, method, url, body, headers):\n        body = self.fixtures.load('.xml')\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n\n    def (self, method, url, body, headers):\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n\n    def (self, method, url, body, headers):\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n\n    def (self, method, url, body, headers):\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n\n    def (self, method, url, body, headers):\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n\n    def (self, method, url, body, headers):\n        return (httplib.OK, body, {}, httplib.responses[httplib.OK])\n    "