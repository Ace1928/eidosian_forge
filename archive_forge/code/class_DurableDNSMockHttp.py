import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
class DurableDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('durabledns')

    def _services_dns_listZones_php(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_listRecords_php(self, method, url, body, headers):
        body = self.fixtures.load('list_records.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_listRecords_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getZone_php(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getZone_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getRecord_php(self, method, url, body, headers):
        body = self.fixtures.load('get_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getRecord_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_record_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getRecord_php_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_record_RECORD_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createZone_php_WITH_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getZone_php_WITH_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_WITH_EXTRA_PARAMS.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createZone_php_NO_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getZone_php_NO_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_NO_EXTRA_PARAMS.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createZone_php_ZONE_ALREADY_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_ZONE_ALREADY_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createRecord_php_NO_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_NO_EXTRA_PARAMS.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createRecord_php_WITH_EXTRA_PARAMS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_WITH_EXTRA_PARAMS.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_createRecord_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('create_record_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_updateZone_php_UPDATE_ZONE(self, method, url, body, headers):
        body = self.fixtures.load('update_zone_UPDATE_ZONE.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_getZone_php_UPDATE_ZONE(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_UPDATE_ZONE.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_updateZone_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('update_zone_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_updateRecord_php(self, method, url, body, headers):
        body = self.fixtures.load('update_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_updateRecord_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('update_record_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_deleteZone_php(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_deleteZone_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_deleteRecord_php(self, method, url, body, headers):
        body = self.fixtures.load('delete_record.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_deleteRecord_php_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_RECORD_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _services_dns_deleteRecord_php_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_ZONE_DOES_NOT_EXIST.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])