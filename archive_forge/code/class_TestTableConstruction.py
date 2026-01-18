from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
class TestTableConstruction(unittest.TestCase):

    def setUp(self):
        self.layer2 = Layer2('access_key', 'secret_key')
        self.api = Mock()
        self.layer2.layer1 = self.api

    def test_get_table(self):
        self.api.describe_table.return_value = DESCRIBE_TABLE
        table = self.layer2.get_table('footest')
        self.assertEqual(table.name, 'footest')
        self.assertEqual(table.create_time, 1353526122.785)
        self.assertEqual(table.status, 'ACTIVE')
        self.assertEqual(table.item_count, 1)
        self.assertEqual(table.size_bytes, 21)
        self.assertEqual(table.read_units, 5)
        self.assertEqual(table.write_units, 5)
        self.assertEqual(table.schema, Schema.create(hash_key=('foo', 'N')))

    def test_create_table_without_api_call(self):
        table = self.layer2.table_from_schema(name='footest', schema=Schema.create(hash_key=('foo', 'N')))
        self.assertEqual(table.name, 'footest')
        self.assertEqual(table.schema, Schema.create(hash_key=('foo', 'N')))
        self.assertEqual(self.api.describe_table.call_count, 0)

    def test_create_schema_with_hash_and_range(self):
        schema = self.layer2.create_schema('foo', int, 'bar', str)
        self.assertEqual(schema.hash_key_name, 'foo')
        self.assertEqual(schema.hash_key_type, 'N')
        self.assertEqual(schema.range_key_name, 'bar')
        self.assertEqual(schema.range_key_type, 'S')

    def test_create_schema_with_hash(self):
        schema = self.layer2.create_schema('foo', str)
        self.assertEqual(schema.hash_key_name, 'foo')
        self.assertEqual(schema.hash_key_type, 'S')
        self.assertIsNone(schema.range_key_name)
        self.assertIsNone(schema.range_key_type)