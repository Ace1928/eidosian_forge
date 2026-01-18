import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_GOOGLE, DNS_KEYWORD_PARAMS_GOOGLE
from libcloud.common.google import GoogleBaseAuthConnection
from libcloud.dns.drivers.google import GoogleDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
class GoogleTests(GoogleTestCase):

    def setUp(self):
        GoogleDNSMockHttp.test = self
        GoogleDNSDriver.connectionCls.conn_class = GoogleDNSMockHttp
        GoogleBaseAuthConnection.conn_class = GoogleAuthMockHttp
        GoogleDNSMockHttp.type = None
        kwargs = DNS_KEYWORD_PARAMS_GOOGLE.copy()
        kwargs['auth_type'] = 'IA'
        self.driver = GoogleDNSDriver(*DNS_PARAMS_GOOGLE, **kwargs)

    def test_default_scopes(self):
        self.assertIsNone(self.driver.scopes)

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertEqual(len(zones), 2)

    def test_list_records(self):
        zone = self.driver.list_zones()[0]
        records = self.driver.list_records(zone=zone)
        self.assertEqual(len(records), 3)

    def test_get_zone(self):
        zone = self.driver.get_zone('example-com')
        self.assertEqual(zone.id, 'example-com')
        self.assertEqual(zone.domain, 'example.com.')

    def test_get_zone_does_not_exist(self):
        GoogleDNSMockHttp.type = 'ZONE_DOES_NOT_EXIST'
        try:
            self.driver.get_zone('example-com')
        except ZoneDoesNotExistError as e:
            self.assertEqual(e.zone_id, 'example-com')
        else:
            self.fail('Exception not thrown')

    def test_get_record(self):
        GoogleDNSMockHttp.type = 'FILTER_ZONES'
        zone = self.driver.list_zones()[0]
        record = self.driver.get_record(zone.id, 'A:foo.example.com.')
        self.assertEqual(record.id, 'A:foo.example.com.')
        self.assertEqual(record.name, 'foo.example.com.')
        self.assertEqual(record.type, 'A')
        self.assertEqual(record.zone.id, 'example-com')

    def test_get_record_zone_does_not_exist(self):
        GoogleDNSMockHttp.type = 'ZONE_DOES_NOT_EXIST'
        try:
            self.driver.get_record('example-com', 'a:a')
        except ZoneDoesNotExistError as e:
            self.assertEqual(e.zone_id, 'example-com')
        else:
            self.fail('Exception not thrown')

    def test_get_record_record_does_not_exist(self):
        GoogleDNSMockHttp.type = 'RECORD_DOES_NOT_EXIST'
        try:
            self.driver.get_record('example-com', 'A:foo')
        except RecordDoesNotExistError as e:
            self.assertEqual(e.record_id, 'A:foo')
        else:
            self.fail('Exception not thrown')

    def test_create_zone(self):
        extra = {'description': 'new domain for example.org'}
        zone = self.driver.create_zone('example.org.', extra)
        self.assertEqual(zone.domain, 'example.org.')
        self.assertEqual(zone.extra['description'], extra['description'])
        self.assertEqual(len(zone.extra['nameServers']), 4)

    def test_delete_zone(self):
        zone = self.driver.get_zone('example-com')
        res = self.driver.delete_zone(zone)
        self.assertTrue(res)

    def test_ex_bulk_record_changes(self):
        zone = self.driver.get_zone('example-com')
        records = self.driver.ex_bulk_record_changes(zone, {})
        self.assertEqual(records['additions'][0].name, 'foo.example.com.')
        self.assertEqual(records['additions'][0].type, 'A')
        self.assertEqual(records['deletions'][0].name, 'bar.example.com.')
        self.assertEqual(records['deletions'][0].type, 'A')