import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSPOD
from libcloud.dns.drivers.dnspod import DNSPodDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class DNSPodMockHttp(MockHttp):
    fixtures = DNSFileFixtures('dnspod')

    def _Domain_List_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_List_LIST_ZONES(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Info_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (404, body, {}, httplib.responses[httplib.OK])

    def _Domain_Info_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Remove_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Remove_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Create_CREATE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Create_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('zone_already_exists.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_List_LIST_RECORDS_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('list_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Info_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Info_GET_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Remove_DELETE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Remove_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('delete_record_record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Create_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Domain_Info_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Info_CREATE_RECORD_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _Record_Create_RECORD_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('record_already_exists.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])