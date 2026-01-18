import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.dns.drivers.nfsn import NFSNDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
class NFSNTestCase(LibcloudTestCase):

    def setUp(self):
        NFSNDNSDriver.connectionCls.conn_class = NFSNMockHttp
        NFSNMockHttp.type = None
        self.driver = NFSNDNSDriver('testid', 'testsecret')
        self.test_zone = Zone(id='example.com', domain='example.com', driver=self.driver, type='master', ttl=None, extra={})
        self.test_record = Record(id=None, name='', data='192.0.2.1', type=RecordType.A, zone=self.test_zone, driver=self.driver, extra={})

    def test_list_zones(self):
        with self.assertRaises(NotImplementedError):
            self.driver.list_zones()

    def test_create_zone(self):
        with self.assertRaises(NotImplementedError):
            self.driver.create_zone('example.com')

    def test_get_zone(self):
        zone = self.driver.get_zone('example.com')
        self.assertEqual(zone.id, None)
        self.assertEqual(zone.domain, 'example.com')

    def test_delete_zone(self):
        with self.assertRaises(NotImplementedError):
            self.driver.delete_zone(self.test_zone)

    def test_create_record(self):
        NFSNMockHttp.type = 'CREATED'
        record = self.test_zone.create_record(name='newrecord', type=RecordType.A, data='127.0.0.1', extra={'ttl': 900})
        self.assertEqual(record.id, None)
        self.assertEqual(record.name, 'newrecord')
        self.assertEqual(record.data, '127.0.0.1')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 900)

    def test_get_record(self):
        with self.assertRaises(NotImplementedError):
            self.driver.get_record('example.com', '12345')

    def test_delete_record(self):
        self.assertTrue(self.test_record.delete())

    def test_list_records(self):
        records = self.driver.list_records(self.test_zone)
        self.assertEqual(len(records), 2)

    def test_ex_get_records_by(self):
        NFSNMockHttp.type = 'ONE_RECORD'
        records = self.driver.ex_get_records_by(self.test_zone, type=RecordType.A)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.name, '')
        self.assertEqual(record.data, '192.0.2.1')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.ttl, 3600)

    def test_get_zone_not_found(self):
        NFSNMockHttp.type = 'NOT_FOUND'
        with self.assertRaises(ZoneDoesNotExistError):
            self.driver.get_zone('example.com')

    def test_delete_record_not_found(self):
        NFSNMockHttp.type = 'NOT_FOUND'
        with self.assertRaises(RecordDoesNotExistError):
            self.assertTrue(self.test_record.delete())