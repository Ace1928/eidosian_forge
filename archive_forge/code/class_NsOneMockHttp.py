import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.common.nsone import NsOneException
from libcloud.test.secrets import DNS_PARAMS_NSONE
from libcloud.dns.drivers.nsone import NsOneDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class NsOneMockHttp(MockHttp):
    fixtures = DNSFileFixtures('nsone')

    def _v1_zones_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_getzone_com_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_zonedoesnotexist_com_GET_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (404, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_newzone_com_CREATE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_newzone_com_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('zone_already_exists.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_LIST_RECORDS_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_LIST_RECORDS_EMPTY(self, method, url, body, headers):
        body = self.fixtures.load('list_records_empty.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_LIST_RECORDS_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_example_com_A_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('record_does_not_exist.json')
        return (404, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_example_com_A_DELETE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_getrecord_com_getrecord_com_A_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_getrecord_com_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_getrecord_com_getrecord_com_A_GET_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_example_com_test_com_A_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_example_com_test_com_A_CREATE_RECORD_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('create_record_already_exists.json')
        return (404, body, {}, httplib.responses[httplib.OK])

    def _v1_zones_test_com_example_com_test_com_A_CREATE_RECORD_ZONE_NOT_FOUND(self, method, url, body, headers):
        body = self.fixtures.load('create_record_zone_not_found.json')
        return (404, body, {}, httplib.responses[httplib.OK])