import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.powerdns import PowerDNSDriver
class PowerDNSTestCase(LibcloudTestCase):

    def setUp(self):
        PowerDNSDriver.connectionCls.conn_class = PowerDNSMockHttp
        PowerDNSMockHttp.type = None
        self.driver = PowerDNSDriver('testsecret')
        self.test_zone = Zone(id='example.com.', domain='example.com', driver=self.driver, type='master', ttl=None, extra={})
        self.test_record = Record(id=None, name='', data='192.0.2.1', type=RecordType.A, zone=self.test_zone, driver=self.driver, extra={})

    def test_create_record(self):
        record = self.test_zone.create_record(name='newrecord.example.com', type=RecordType.A, data='192.0.5.4', extra={'ttl': 86400})
        self.assertIsNone(record.id)
        self.assertEqual(record.name, 'newrecord.example.com')
        self.assertEqual(record.data, '192.0.5.4')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 86400)

    def test_create_zone(self):
        extra = {'nameservers': ['ns1.example.org', 'ns2.example.org']}
        zone = self.driver.create_zone('example.org', extra=extra)
        self.assertEqual(zone.id, 'example.org.')
        self.assertEqual(zone.domain, 'example.org')
        self.assertIsNone(zone.type)
        self.assertIsNone(zone.ttl)

    def test_delete_record(self):
        self.assertTrue(self.test_record.delete())

    def test_delete_zone(self):
        self.assertTrue(self.test_zone.delete())

    def test_get_record(self):
        with self.assertRaises(NotImplementedError):
            self.driver.get_record('example.com.', '12345')

    def test_get_zone(self):
        zone = self.driver.get_zone('example.com.')
        self.assertEqual(zone.id, 'example.com.')
        self.assertEqual(zone.domain, 'example.com')
        self.assertIsNone(zone.type)
        self.assertIsNone(zone.ttl)

    def test_list_record_types(self):
        result = self.driver.list_record_types()
        self.assertEqual(len(result), 23)

    def test_list_records(self):
        records = self.driver.list_records(self.test_zone)
        self.assertEqual(len(records), 4)

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertEqual(zones[0].id, 'example.com.')
        self.assertEqual(zones[0].domain, 'example.com')
        self.assertIsNone(zones[0].type)
        self.assertIsNone(zones[0].ttl)
        self.assertEqual(zones[1].id, 'example.net.')
        self.assertEqual(zones[1].domain, 'example.net')
        self.assertIsNone(zones[1].type)
        self.assertIsNone(zones[1].ttl)

    def test_update_record(self):
        record = self.driver.update_record(self.test_record, name='newrecord.example.com', type=RecordType.A, data='127.0.0.1', extra={'ttl': 300})
        self.assertIsNone(record.id)
        self.assertEqual(record.name, 'newrecord.example.com')
        self.assertEqual(record.data, '127.0.0.1')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 300)

    def test_update_zone(self):
        with self.assertRaises(NotImplementedError):
            self.driver.update_zone(self.test_zone, 'example.net')

    def test_create_existing_zone(self):
        PowerDNSMockHttp.type = 'EXISTS'
        extra = {'nameservers': ['ns1.example.com', 'ns2.example.com']}
        with self.assertRaises(ZoneAlreadyExistsError):
            self.driver.create_zone('example.com', extra=extra)

    def test_get_missing_zone(self):
        PowerDNSMockHttp.type = 'MISSING'
        with self.assertRaises(ZoneDoesNotExistError):
            self.driver.get_zone('example.com.')

    def test_delete_missing_record(self):
        PowerDNSMockHttp.type = 'MISSING'
        self.assertFalse(self.test_record.delete())

    def test_delete_missing_zone(self):
        PowerDNSMockHttp.type = 'MISSING'
        self.assertFalse(self.test_zone.delete())