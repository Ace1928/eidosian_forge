import sys
import json
import unittest
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI_LIVE
from libcloud.common.gandi_live import JsonParseError, GandiLiveBaseError, InvalidRequestError
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.gandi_live import GandiLiveDNSDriver
from libcloud.test.common.test_gandi_live import BaseGandiLiveMockHttp
class GandiLiveTests(unittest.TestCase):

    def setUp(self):
        GandiLiveDNSDriver.connectionCls.conn_class = GandiLiveMockHttp
        GandiLiveMockHttp.type = None
        self.driver = GandiLiveDNSDriver(*DNS_GANDI_LIVE)
        self.test_zone = Zone(id='example.com', type='master', ttl=None, domain='example.com', extra={'zone_uuid': 'a53re'}, driver=self)
        self.test_bad_zone = Zone(id='badexample.com', type='master', ttl=None, domain='badexample.com', extra={'zone_uuid': 'a53rf'}, driver=self)
        self.test_record = Record(id='A:bob', type=RecordType.A, name='bob', zone=self.test_zone, data='127.0.0.1', driver=self, extra={})
        self.test_bad_record = Record(id='A:jane', type=RecordType.A, name='jane', zone=self.test_bad_zone, data='127.0.0.1', driver=self, extra={})

    def test_list_zones(self):
        zones = self.driver.list_zones()
        self.assertEqual(len(zones), 2)
        zone = zones[0]
        self.assertEqual(zone.id, 'example.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'example.com')
        self.assertIsNone(zone.ttl)
        zone = zones[1]
        self.assertEqual(zone.id, 'example.net')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'example.net')
        self.assertIsNone(zone.ttl)

    def test_create_zone(self):
        zone = self.driver.create_zone('example.org', extra={'name': 'Example'})
        self.assertEqual(zone.id, 'example.org')
        self.assertEqual(zone.domain, 'example.org')
        self.assertEqual(zone.extra['zone_uuid'], '54321')

    def test_create_zone_without_name(self):
        zone = self.driver.create_zone('example.org')
        self.assertEqual(zone.id, 'example.org')
        self.assertEqual(zone.domain, 'example.org')
        self.assertEqual(zone.extra['zone_uuid'], '54321')

    def test_get_zone(self):
        zone = self.driver.get_zone('example.com')
        self.assertEqual(zone.id, 'example.com')
        self.assertEqual(zone.type, 'master')
        self.assertEqual(zone.domain, 'example.com')
        self.assertIsNone(zone.ttl)

    def test_list_records(self):
        records = self.driver.list_records(self.test_zone)
        self.assertEqual(len(records), 3)
        record = records[0]
        self.assertEqual(record.id, 'A:@')
        self.assertEqual(record.name, '@')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '127.0.0.1')
        record = records[1]
        self.assertEqual(record.id, 'CNAME:www')
        self.assertEqual(record.name, 'www')
        self.assertEqual(record.type, RecordType.CNAME)
        self.assertEqual(record.data, 'bob.example.com.')
        record = records[2]
        self.assertEqual(record.id, 'A:bob')
        self.assertEqual(record.name, 'bob')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '127.0.1.1')

    def test_get_record(self):
        record = self.driver.get_record(self.test_zone.id, 'A:bob')
        self.assertEqual(record.id, 'A:bob')
        self.assertEqual(record.name, 'bob')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '127.0.1.1')

    def test_create_record(self):
        record = self.driver.create_record('alice', self.test_zone, 'AAAA', '::1', extra={'ttl': 400})
        self.assertEqual(record.id, 'AAAA:alice')
        self.assertEqual(record.name, 'alice')
        self.assertEqual(record.type, RecordType.AAAA)
        self.assertEqual(record.data, '::1')

    def test_create_record_doesnt_throw_if_ttl_is_not_provided(self):
        record = self.driver.create_record('alice', self.test_zone, 'AAAA', '::1')
        self.assertEqual(record.id, 'AAAA:alice')
        self.assertEqual(record.name, 'alice')
        self.assertEqual(record.type, RecordType.AAAA)
        self.assertEqual(record.data, '::1')

    def test_bad_record_validation(self):
        with self.assertRaises(RecordError) as ctx:
            self.driver.create_record('alice', self.test_zone, 'AAAA', '1' * 1025, extra={'ttl': 400})
        self.assertTrue('Record data must be' in str(ctx.exception))
        with self.assertRaises(RecordError) as ctx:
            self.driver.create_record('alice', self.test_zone, 'AAAA', '::1', extra={'ttl': 10})
        self.assertTrue('TTL must be at least' in str(ctx.exception))
        with self.assertRaises(RecordError) as ctx:
            self.driver.create_record('alice', self.test_zone, 'AAAA', '::1', extra={'ttl': 31 * 24 * 60 * 60})
        self.assertTrue('TTL must not exceed' in str(ctx.exception))

    def test_update_record(self):
        record = self.driver.update_record(self.test_record, 'bob', RecordType.A, '192.168.0.2', {'ttl': 500})
        self.assertEqual(record.id, 'A:bob')
        self.assertEqual(record.name, 'bob')
        self.assertEqual(record.type, RecordType.A)
        self.assertEqual(record.data, '192.168.0.2')

    def test_delete_record(self):
        success = self.driver.delete_record(self.test_record)
        self.assertTrue(success)

    def test_export_bind(self):
        bind_export = self.driver.export_zone_to_bind_format(self.test_zone)
        bind_lines = bind_export.decode('utf8').split('\n')
        self.assertEqual(bind_lines[0], '@ 10800 IN A 127.0.0.1')

    def test_bad_json_response(self):
        with self.assertRaises(JsonParseError):
            self.driver.get_zone('badexample.com')

    def test_no_record_found(self):
        with self.assertRaises(RecordDoesNotExistError):
            self.driver.get_record(self.test_zone.id, 'A:none')

    def test_record_already_exists(self):
        with self.assertRaises(RecordAlreadyExistsError):
            self.driver.create_record('bob', self.test_bad_zone, 'A', '127.0.0.1', extra={'ttl': 400})

    def test_no_zone_found(self):
        with self.assertRaises(ZoneDoesNotExistError):
            self.driver.get_zone('nosuchzone.com')

    def test_zone_already_exists(self):
        with self.assertRaises(ZoneAlreadyExistsError):
            self.driver.create_zone('badexample.com')

    def test_suberrors(self):
        with self.assertRaises(InvalidRequestError) as ctx:
            self.driver.update_record(self.test_bad_record, 'jane', RecordType.A, '192.168.0.2', {'ttl': 500})
        self.assertTrue('is not a foo' in str(ctx.exception))

    def test_other_error(self):
        with self.assertRaises(GandiLiveBaseError):
            self.driver.list_records(self.test_bad_zone)

    def test_mx_record(self):
        record = self.driver.get_record(self.test_zone.id, 'MX:lists')
        self.assertEqual(record.extra['priority'], '10')
        self.assertTrue('_other_records' in record.extra)
        other_record = record.extra['_other_records'][0]
        self.assertEqual(other_record['extra']['priority'], '20')

    def test_ex_create_multivalue_record(self):
        records = self.driver.ex_create_multi_value_record('alice', self.test_zone, 'AAAA', ['::1', '::2'], extra={'ttl': 400})
        self.assertEqual(records[0].id, 'AAAA:alice')
        self.assertEqual(records[0].name, 'alice')
        self.assertEqual(records[0].type, RecordType.AAAA)
        self.assertEqual(records[0].data, '::1')
        self.assertEqual(records[1].id, 'AAAA:alice')
        self.assertEqual(records[1].name, 'alice')
        self.assertEqual(records[1].type, RecordType.AAAA)
        self.assertEqual(records[1].data, '::2')

    def test_update_multivalue_record(self):
        record = self.driver.get_record(self.test_zone.id, 'MX:lists')
        updated = self.driver.update_record(record, None, None, 'mail1', {'ttl': 400, 'priority': 10})
        self.assertEqual(updated.extra['priority'], '10')
        self.assertEqual(updated.data, 'mail1')
        self.assertTrue('_other_records' in record.extra)
        other_record = record.extra['_other_records'][0]
        self.assertEqual(other_record['extra']['priority'], '20')

    def test_ex_update_gandi_zone_name(self):
        updated = self.driver.ex_update_gandi_zone_name('111111', 'Foo')
        self.assertTrue(updated)

    def test_ex_delete_gandi_zone(self):
        deleted = self.driver.ex_delete_gandi_zone('111111')
        self.assertTrue(deleted)