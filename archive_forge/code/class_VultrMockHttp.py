import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.dns.drivers.vultr import VultrDNSDriver, VultrDNSDriverV1
from libcloud.test.file_fixtures import DNSFileFixtures
class VultrMockHttp(MockHttp):
    fixtures = DNSFileFixtures('vultr')

    def _v1_dns_list(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records(self, method, url, body, headers):
        body = self.fixtures.load('list_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_GET_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_EMPTY_RECORDS_LIST(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records_EMPTY_RECORDS_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_records_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_GET_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records_GET_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_GET_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records_GET_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_delete_domain(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_delete_record(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_create_domain(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_create_domain_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_delete_domain_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_delete_record_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_DELETE_RECORD_RECORD_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('test_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_list_LIST_RECORDS_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_dns_records_LIST_RECORDS_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])