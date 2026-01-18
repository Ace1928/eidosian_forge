from tests.compat import mock, unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import (STRING, NUMBER, BINARY,
from boto.exception import JSONResponseError
from boto.compat import six, long_type
class ItemTestCase(unittest.TestCase):
    if six.PY2:
        assertCountEqual = unittest.TestCase.assertItemsEqual

    def setUp(self):
        super(ItemTestCase, self).setUp()
        self.table = Table('whatever', connection=FakeDynamoDBConnection())
        self.johndoe = self.create_item({'username': 'johndoe', 'first_name': 'John', 'date_joined': 12345})

    def create_item(self, data):
        return Item(self.table, data=data)

    def test_initialization(self):
        empty_item = Item(self.table)
        self.assertEqual(empty_item.table, self.table)
        self.assertEqual(empty_item._data, {})
        full_item = Item(self.table, data={'username': 'johndoe', 'date_joined': 12345})
        self.assertEqual(full_item.table, self.table)
        self.assertEqual(full_item._data, {'username': 'johndoe', 'date_joined': 12345})

    def test_keys(self):
        self.assertCountEqual(self.johndoe.keys(), ['date_joined', 'first_name', 'username'])

    def test_values(self):
        self.assertCountEqual(self.johndoe.values(), [12345, 'John', 'johndoe'])

    def test_contains(self):
        self.assertIn('username', self.johndoe)
        self.assertIn('first_name', self.johndoe)
        self.assertIn('date_joined', self.johndoe)
        self.assertNotIn('whatever', self.johndoe)

    def test_iter(self):
        self.assertCountEqual(self.johndoe, ['johndoe', 'John', 12345])

    def test_get(self):
        self.assertEqual(self.johndoe.get('username'), 'johndoe')
        self.assertEqual(self.johndoe.get('first_name'), 'John')
        self.assertEqual(self.johndoe.get('date_joined'), 12345)
        self.assertEqual(self.johndoe.get('last_name'), None)
        self.assertEqual(self.johndoe.get('last_name', True), True)

    def test_items(self):
        self.assertCountEqual(self.johndoe.items(), [('date_joined', 12345), ('first_name', 'John'), ('username', 'johndoe')])

    def test_attribute_access(self):
        self.assertEqual(self.johndoe['username'], 'johndoe')
        self.assertEqual(self.johndoe['first_name'], 'John')
        self.assertEqual(self.johndoe['date_joined'], 12345)
        self.assertEqual(self.johndoe['last_name'], None)
        self.johndoe['last_name'] = 'Doe'
        self.assertEqual(self.johndoe['last_name'], 'Doe')
        del self.johndoe['last_name']
        self.assertEqual(self.johndoe['last_name'], None)

    def test_needs_save(self):
        self.johndoe.mark_clean()
        self.assertFalse(self.johndoe.needs_save())
        self.johndoe['last_name'] = 'Doe'
        self.assertTrue(self.johndoe.needs_save())

    def test_needs_save_set_changed(self):
        self.johndoe.mark_clean()
        self.assertFalse(self.johndoe.needs_save())
        self.johndoe['friends'] = set(['jane', 'alice'])
        self.assertTrue(self.johndoe.needs_save())
        self.johndoe.mark_clean()
        self.assertFalse(self.johndoe.needs_save())
        self.johndoe['friends'].add('bob')
        self.assertTrue(self.johndoe.needs_save())

    def test_mark_clean(self):
        self.johndoe['last_name'] = 'Doe'
        self.assertTrue(self.johndoe.needs_save())
        self.johndoe.mark_clean()
        self.assertFalse(self.johndoe.needs_save())

    def test_load(self):
        empty_item = Item(self.table)
        empty_item.load({'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}})
        self.assertEqual(empty_item['username'], 'johndoe')
        self.assertEqual(empty_item['date_joined'], 1366056668)
        self.assertEqual(sorted(empty_item['friends']), sorted(['alice', 'bob', 'jane']))

    def test_get_keys(self):
        self.table.schema = [HashKey('username'), RangeKey('date_joined')]
        self.assertEqual(self.johndoe.get_keys(), {'username': 'johndoe', 'date_joined': 12345})

    def test_get_raw_keys(self):
        self.table.schema = [HashKey('username'), RangeKey('date_joined')]
        self.assertEqual(self.johndoe.get_raw_keys(), {'username': {'S': 'johndoe'}, 'date_joined': {'N': '12345'}})

    def test_build_expects(self):
        self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': False}, 'username': {'Exists': False}, 'date_joined': {'Exists': False}})
        self.johndoe.mark_clean()
        self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'username': {'Exists': True, 'Value': {'S': 'johndoe'}}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})
        self.johndoe['first_name'] = 'Johann'
        self.johndoe['last_name'] = 'Doe'
        del self.johndoe['date_joined']
        self.assertEqual(self.johndoe.build_expects(), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'last_name': {'Exists': False}, 'username': {'Exists': True, 'Value': {'S': 'johndoe'}}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})
        self.assertEqual(self.johndoe.build_expects(fields=['first_name', 'last_name', 'date_joined']), {'first_name': {'Exists': True, 'Value': {'S': 'John'}}, 'last_name': {'Exists': False}, 'date_joined': {'Exists': True, 'Value': {'N': '12345'}}})

    def test_prepare_full(self):
        self.assertEqual(self.johndoe.prepare_full(), {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'date_joined': {'N': '12345'}})
        self.johndoe['friends'] = set(['jane', 'alice'])
        data = self.johndoe.prepare_full()
        self.assertEqual(data['username'], {'S': 'johndoe'})
        self.assertEqual(data['first_name'], {'S': 'John'})
        self.assertEqual(data['date_joined'], {'N': '12345'})
        self.assertCountEqual(data['friends']['SS'], ['jane', 'alice'])

    def test_prepare_full_empty_set(self):
        self.johndoe['friends'] = set()
        self.assertEqual(self.johndoe.prepare_full(), {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'date_joined': {'N': '12345'}})

    def test_prepare_partial(self):
        self.johndoe.mark_clean()
        self.johndoe['first_name'] = 'Johann'
        self.johndoe['last_name'] = 'Doe'
        del self.johndoe['date_joined']
        final_data, fields = self.johndoe.prepare_partial()
        self.assertEqual(final_data, {'date_joined': {'Action': 'DELETE'}, 'first_name': {'Action': 'PUT', 'Value': {'S': 'Johann'}}, 'last_name': {'Action': 'PUT', 'Value': {'S': 'Doe'}}})
        self.assertEqual(fields, set(['first_name', 'last_name', 'date_joined']))

    def test_prepare_partial_empty_set(self):
        self.johndoe.mark_clean()
        self.johndoe['first_name'] = 'Johann'
        self.johndoe['last_name'] = 'Doe'
        del self.johndoe['date_joined']
        self.johndoe['friends'] = set()
        final_data, fields = self.johndoe.prepare_partial()
        self.assertEqual(final_data, {'date_joined': {'Action': 'DELETE'}, 'first_name': {'Action': 'PUT', 'Value': {'S': 'Johann'}}, 'last_name': {'Action': 'PUT', 'Value': {'S': 'Doe'}}})
        self.assertEqual(fields, set(['first_name', 'last_name', 'date_joined']))

    def test_save_no_changes(self):
        with mock.patch.object(self.table, '_put_item', return_value=True) as mock_put_item:
            self.johndoe.mark_clean()
            self.assertFalse(self.johndoe.save())
        self.assertFalse(mock_put_item.called)

    def test_save_with_changes(self):
        with mock.patch.object(self.table, '_put_item', return_value=True) as mock_put_item:
            self.johndoe.mark_clean()
            self.johndoe['first_name'] = 'J'
            self.johndoe['new_attr'] = 'never_seen_before'
            self.assertTrue(self.johndoe.save())
            self.assertFalse(self.johndoe.needs_save())
        self.assertTrue(mock_put_item.called)
        mock_put_item.assert_called_once_with({'username': {'S': 'johndoe'}, 'first_name': {'S': 'J'}, 'new_attr': {'S': 'never_seen_before'}, 'date_joined': {'N': '12345'}}, expects={'username': {'Value': {'S': 'johndoe'}, 'Exists': True}, 'first_name': {'Value': {'S': 'John'}, 'Exists': True}, 'new_attr': {'Exists': False}, 'date_joined': {'Value': {'N': '12345'}, 'Exists': True}})

    def test_save_with_changes_overwrite(self):
        with mock.patch.object(self.table, '_put_item', return_value=True) as mock_put_item:
            self.johndoe['first_name'] = 'J'
            self.johndoe['new_attr'] = 'never_seen_before'
            self.assertTrue(self.johndoe.save(overwrite=True))
            self.assertFalse(self.johndoe.needs_save())
        self.assertTrue(mock_put_item.called)
        mock_put_item.assert_called_once_with({'username': {'S': 'johndoe'}, 'first_name': {'S': 'J'}, 'new_attr': {'S': 'never_seen_before'}, 'date_joined': {'N': '12345'}}, expects=None)

    def test_partial_no_changes(self):
        with mock.patch.object(self.table, '_update_item', return_value=True) as mock_update_item:
            self.johndoe.mark_clean()
            self.assertFalse(self.johndoe.partial_save())
        self.assertFalse(mock_update_item.called)

    def test_partial_with_changes(self):
        self.table.schema = [HashKey('username')]
        with mock.patch.object(self.table, '_update_item', return_value=True) as mock_update_item:
            self.johndoe.mark_clean()
            self.johndoe['first_name'] = 'J'
            self.johndoe['last_name'] = 'Doe'
            del self.johndoe['date_joined']
            self.assertTrue(self.johndoe.partial_save())
            self.assertFalse(self.johndoe.needs_save())
        self.assertTrue(mock_update_item.called)
        mock_update_item.assert_called_once_with({'username': 'johndoe'}, {'first_name': {'Action': 'PUT', 'Value': {'S': 'J'}}, 'last_name': {'Action': 'PUT', 'Value': {'S': 'Doe'}}, 'date_joined': {'Action': 'DELETE'}}, expects={'first_name': {'Value': {'S': 'John'}, 'Exists': True}, 'last_name': {'Exists': False}, 'date_joined': {'Value': {'N': '12345'}, 'Exists': True}})

    def test_delete(self):
        self.table.schema = [HashKey('username'), RangeKey('date_joined')]
        with mock.patch.object(self.table, 'delete_item', return_value=True) as mock_delete_item:
            self.johndoe.delete()
        self.assertTrue(mock_delete_item.called)
        mock_delete_item.assert_called_once_with(username='johndoe', date_joined=12345)

    def test_nonzero(self):
        self.assertTrue(self.johndoe)
        self.assertFalse(self.create_item({}))