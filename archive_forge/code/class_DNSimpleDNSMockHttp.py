import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DNSIMPLE
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.dnsimple import DNSimpleDNSDriver
class DNSimpleDNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('dnsimple')

    def _v1_domains(self, method, url, body, headers):
        body = self.fixtures.load('list_domains.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('create_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1(self, method, url, body, headers):
        body = self.fixtures.load('get_domain.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1_records(self, method, url, body, headers):
        body = self.fixtures.load('list_domain_records.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1_records_123(self, method, url, body, headers):
        body = self.fixtures.load('get_domain_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1_records_CREATE(self, method, url, body, headers):
        body = self.fixtures.load('create_domain_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1_records_123_UPDATE(self, method, url, body, headers):
        body = self.fixtures.load('update_domain_record.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _v1_domains_1_DELETE_200(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _v1_domains_1_DELETE_204(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.NO_CONTENT])

    def _v1_domains_1_records_2_DELETE_200(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.OK])

    def _v1_domains_1_records_2_DELETE_204(self, method, url, body, headers):
        return (httplib.OK, '', {}, httplib.responses[httplib.NO_CONTENT])