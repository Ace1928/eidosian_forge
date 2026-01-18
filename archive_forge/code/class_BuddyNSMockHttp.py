import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.base import Zone
from libcloud.dns.types import ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_BUDDYNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.buddyns import BuddyNSDNSDriver
class BuddyNSMockHttp(MockHttp):
    fixtures = DNSFileFixtures('buddyns')

    def _api_v2_zone_EMPTY_ZONES_LIST(self, method, url, body, headers):
        body = self.fixtures.load('empty_zones_list.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_LIST_ZONES(self, method, url, body, headers):
        body = self.fixtures.load('list_zones.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_zonedoesnotexist_com_GET_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (404, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_myexample_com_GET_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('get_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_test_com_DELETE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('delete_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_test_com_DELETE_ZONE_ZONE_DOES_NOT_EXIST(self, method, url, body, headers):
        body = self.fixtures.load('zone_does_not_exist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_CREATE_ZONE_SUCCESS(self, method, url, body, headers):
        body = self.fixtures.load('create_zone_success.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _api_v2_zone_CREATE_ZONE_ZONE_ALREADY_EXISTS(self, method, url, body, headers):
        body = self.fixtures.load('zone_already_exists.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])