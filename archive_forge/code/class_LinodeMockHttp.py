import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_LINODE, DNS_KEYWORD_PARAMS_LINODE
from libcloud.common.linode import LinodeException
from libcloud.dns.drivers.linode import LinodeDNSDriver, LinodeDNSDriverV3
from libcloud.test.file_fixtures import DNSFileFixtures
class LinodeMockHttp(MockHttp):
    fixtures = DNSFileFixtures('linode')

    def _domain_list(self, method, url, body, headers):
        body = self.fixtures.load('domain_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_resource_list(self, method, url, body, headers):
        body = self.fixtures.load('resource_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ZONE_DOES_NOT_EXIST_domain_resource_list(self, method, url, body, headers):
        body = self.fixtures.load('resource_list_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_ZONE_domain_list(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_ZONE_DOES_NOT_EXIST_domain_list(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_domain_list(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_domain_resource_list(self, method, url, body, headers):
        body = self.fixtures.load('get_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_ZONE_DOES_NOT_EXIST_domain_list(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_ZONE_DOES_NOT_EXIST_domain_resource_list(self, method, url, body, headers):
        body = self.fixtures.load('get_record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_RECORD_DOES_NOT_EXIST_domain_list(self, method, url, body, headers):
        body = self.fixtures.load('get_zone.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GET_RECORD_RECORD_DOES_NOT_EXIST_domain_resource_list(self, method, url, body, headers):
        body = self.fixtures.load('get_record_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_create(self, method, url, body, headers):
        body = self.fixtures.load('create_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _VALIDATION_ERROR_domain_create(self, method, url, body, headers):
        body = self.fixtures.load('create_domain_validation_error.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_update(self, method, url, body, headers):
        body = self.fixtures.load('update_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_resource_create(self, method, url, body, headers):
        body = self.fixtures.load('create_resource.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_resource_update(self, method, url, body, headers):
        body = self.fixtures.load('update_resource.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ZONE_DOES_NOT_EXIST_domain_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_domain_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _domain_resource_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_resource.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _RECORD_DOES_NOT_EXIST_domain_resource_delete(self, method, url, body, headers):
        body = self.fixtures.load('delete_resource_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])