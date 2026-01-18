import sys
import unittest
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI
from libcloud.dns.drivers.gandi import GandiDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_gandi import BaseGandiMockHttp
class GandiMockHttp(BaseGandiMockHttp):
    fixtures = DNSFileFixtures('gandi')

    def _xmlrpc__domain_zone_create(self, method, url, body, headers):
        body = self.fixtures.load('create_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_update(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_list(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_list(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_add(self, method, url, body, headers):
        body = self.fixtures.load('create_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_info(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_update(self, method, url, body, headers):
        body = self.fixtures.load('create_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_version_new(self, method, url, body, headers):
        body = self.fixtures.load('new_version.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_version_set(self, method, url, body, headers):
        body = self.fixtures.load('new_version.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_list_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_doesnt_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_info_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_doesnt_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_list_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_doesnt_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_delete_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_doesnt_exist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_list_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records_empty.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_info_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_record_delete_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_doesnotexist.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_version_new_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('new_version.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _xmlrpc__domain_zone_version_set_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('new_version.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])