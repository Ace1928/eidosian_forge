import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_WORLDWIDEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.common.worldwidedns import InvalidDomainName, NonExistentDomain
from libcloud.dns.drivers.worldwidedns import WorldWideDNSError, WorldWideDNSDriver
class WorldWideDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('worldwidedns')

    def _api_dns_list_asp(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        return (httplib.OK, '405', {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_GET_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _api_dns_new_domain_asp(self, method, url, body, headers):
        return (httplib.OK, '200', {}, httplib.responses[httplib.OK])

    def _api_dns_new_domain_asp_VALIDATION_ERROR(self, method, url, body, headers):
        return (httplib.OK, '410', {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_UPDATE_ZONE(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_UPDATE_ZONE(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_UPDATE_ZONE(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_UPDATE_ZONE')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_CREATE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_CREATE_SECOND_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_CREATE_RECORD(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_CREATE_SECOND_RECORD(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_CREATE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_CREATE_RECORD')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_CREATE_SECOND_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_CREATE_SECOND_RECORD')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_CREATE_RECORD_MAX_ENTRIES(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_CREATE_RECORD_MAX_ENTRIES')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_CREATE_RECORD_MAX_ENTRIES(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_CREATE_RECORD_MAX_ENTRIES(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_CREATE_RECORD_MAX_ENTRIES_WITH_ENTRY(self, method, url, body, headers):
        body = self.fixtures.load('_api_dns_modify_asp_CREATE_RECORD_MAX_ENTRIES_WITH_ENTRY')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_CREATE_RECORD_MAX_ENTRIES_WITH_ENTRY(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_CREATE_RECORD_MAX_ENTRIES_WITH_ENTRY(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_UPDATE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_UPDATE_RECORD(self, method, url, body, headers):
        return (httplib.OK, '211\r\n212\r\n213', {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_UPDATE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_UPDATE_RECORD')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_delete_domain_asp(self, method, url, body, headers):
        return (httplib.OK, '200', {}, httplib.responses[httplib.OK])

    def _api_dns_delete_domain_asp_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        return (httplib.OK, '405', {}, httplib.responses[httplib.OK])

    def _api_dns_list_asp_DELETE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_dns_modify_asp_DELETE_RECORD(self, method, url, body, headers):
        return (httplib.OK, '200', {}, httplib.responses[httplib.OK])

    def _api_dns_list_domain_asp_DELETE_RECORD(self, method, url, body, headers):
        body = self.fixtures.load('api_dns_list_domain_asp_DELETE_RECORD')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])