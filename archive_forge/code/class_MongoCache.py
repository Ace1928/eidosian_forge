import collections.abc
import copy
import functools
from dogpile.cache import region as dp_region
from oslo_utils import uuidutils
from oslo_cache.backends import mongo
from oslo_cache import core
from oslo_cache import exception
from oslo_cache.tests import test_cache
class MongoCache(test_cache.BaseTestCase):

    def setUp(self):
        super(MongoCache, self).setUp()
        global COLLECTIONS
        COLLECTIONS = {}
        mongo.MongoApi._DB = {}
        mongo.MongoApi._MONGO_COLLS = {}
        pymongo_override()
        self.arguments = {'db_hosts': 'localhost:27017', 'db_name': 'ks_cache', 'cache_collection': 'cache', 'username': 'test_user', 'password': 'test_password'}

    def test_missing_db_hosts(self):
        self.arguments.pop('db_hosts')
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_missing_db_name(self):
        self.arguments.pop('db_name')
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_missing_cache_collection_name(self):
        self.arguments.pop('cache_collection')
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_incorrect_write_concern(self):
        self.arguments['w'] = 'one value'
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_correct_write_concern(self):
        self.arguments['w'] = 1
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue10')
        self.assertEqual(1, region.backend.api.w)

    def test_incorrect_read_preference(self):
        self.arguments['read_preference'] = 'inValidValue'
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertEqual('inValidValue', region.backend.api.read_preference)
        random_key = uuidutils.generate_uuid(dashed=False)
        self.assertRaises(ValueError, region.set, random_key, 'dummyValue10')

    def test_correct_read_preference(self):
        self.arguments['read_preference'] = 'secondaryPreferred'
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertEqual('secondaryPreferred', region.backend.api.read_preference)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue10')
        self.assertEqual(3, region.backend.api.read_preference)

    def test_missing_replica_set_name(self):
        self.arguments['use_replica'] = True
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_provided_replica_set_name(self):
        self.arguments['use_replica'] = True
        self.arguments['replicaset_name'] = 'my_replica'
        dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertTrue(True)

    def test_incorrect_mongo_ttl_seconds(self):
        self.arguments['mongo_ttl_seconds'] = 'sixty'
        region = dp_region.make_region()
        self.assertRaises(exception.ConfigurationError, region.configure, 'oslo_cache.mongo', arguments=self.arguments)

    def test_cache_configuration_values_assertion(self):
        self.arguments['use_replica'] = True
        self.arguments['replicaset_name'] = 'my_replica'
        self.arguments['mongo_ttl_seconds'] = 60
        self.arguments['ssl'] = False
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertEqual('localhost:27017', region.backend.api.hosts)
        self.assertEqual('ks_cache', region.backend.api.db_name)
        self.assertEqual('cache', region.backend.api.cache_collection)
        self.assertEqual('test_user', region.backend.api.username)
        self.assertEqual('test_password', region.backend.api.password)
        self.assertEqual(True, region.backend.api.use_replica)
        self.assertEqual('my_replica', region.backend.api.replicaset_name)
        self.assertEqual(False, region.backend.api.conn_kwargs['ssl'])
        self.assertEqual(60, region.backend.api.ttl_seconds)

    def test_multiple_region_cache_configuration(self):
        arguments1 = copy.copy(self.arguments)
        arguments1['cache_collection'] = 'cache_region1'
        region1 = dp_region.make_region().configure('oslo_cache.mongo', arguments=arguments1)
        self.assertEqual('localhost:27017', region1.backend.api.hosts)
        self.assertEqual('ks_cache', region1.backend.api.db_name)
        self.assertEqual('cache_region1', region1.backend.api.cache_collection)
        self.assertEqual('test_user', region1.backend.api.username)
        self.assertEqual('test_password', region1.backend.api.password)
        self.assertIsNone(region1.backend.api._data_manipulator)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        region1.set(random_key1, 'dummyValue10')
        self.assertEqual('dummyValue10', region1.get(random_key1))
        self.assertIsInstance(region1.backend.api._data_manipulator, mongo.BaseTransform)
        class_name = '%s.%s' % (MyTransformer.__module__, 'MyTransformer')
        arguments2 = copy.copy(self.arguments)
        arguments2['cache_collection'] = 'cache_region2'
        arguments2['son_manipulator'] = class_name
        region2 = dp_region.make_region().configure('oslo_cache.mongo', arguments=arguments2)
        self.assertEqual('localhost:27017', region2.backend.api.hosts)
        self.assertEqual('ks_cache', region2.backend.api.db_name)
        self.assertEqual('cache_region2', region2.backend.api.cache_collection)
        self.assertIsNone(region2.backend.api._data_manipulator)
        random_key = uuidutils.generate_uuid(dashed=False)
        region2.set(random_key, 'dummyValue20')
        self.assertEqual('dummyValue20', region2.get(random_key))
        self.assertIsInstance(region2.backend.api._data_manipulator, MyTransformer)
        region1.set(random_key1, 'dummyValue22')
        self.assertEqual('dummyValue22', region1.get(random_key1))

    def test_typical_configuration(self):
        dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertTrue(True)

    def test_backend_get_missing_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        self.assertEqual(NO_VALUE, region.get(random_key))

    def test_backend_set_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue')
        self.assertEqual('dummyValue', region.get(random_key))

    def test_backend_set_data_with_string_as_valid_ttl(self):
        self.arguments['mongo_ttl_seconds'] = '3600'
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertEqual(3600, region.backend.api.ttl_seconds)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue')
        self.assertEqual('dummyValue', region.get(random_key))

    def test_backend_set_data_with_int_as_valid_ttl(self):
        self.arguments['mongo_ttl_seconds'] = 1800
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        self.assertEqual(1800, region.backend.api.ttl_seconds)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue')
        self.assertEqual('dummyValue', region.get(random_key))

    def test_backend_set_none_as_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, None)
        self.assertIsNone(region.get(random_key))

    def test_backend_set_blank_as_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, '')
        self.assertEqual('', region.get(random_key))

    def test_backend_set_same_key_multiple_times(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue')
        self.assertEqual('dummyValue', region.get(random_key))
        dict_value = {'key1': 'value1'}
        region.set(random_key, dict_value)
        self.assertEqual(dict_value, region.get(random_key))
        region.set(random_key, 'dummyValue2')
        self.assertEqual('dummyValue2', region.get(random_key))

    def test_backend_multi_set_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        random_key2 = uuidutils.generate_uuid(dashed=False)
        random_key3 = uuidutils.generate_uuid(dashed=False)
        mapping = {random_key1: 'dummyValue1', random_key2: 'dummyValue2', random_key3: 'dummyValue3'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertFalse(region.get(random_key))
        self.assertEqual('dummyValue1', region.get(random_key1))
        self.assertEqual('dummyValue2', region.get(random_key2))
        self.assertEqual('dummyValue3', region.get(random_key3))

    def test_backend_multi_get_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        random_key2 = uuidutils.generate_uuid(dashed=False)
        random_key3 = uuidutils.generate_uuid(dashed=False)
        mapping = {random_key1: 'dummyValue1', random_key2: '', random_key3: 'dummyValue3'}
        region.set_multi(mapping)
        keys = [random_key, random_key1, random_key2, random_key3]
        results = region.get_multi(keys)
        self.assertEqual(NO_VALUE, results[0])
        self.assertEqual('dummyValue1', results[1])
        self.assertEqual('', results[2])
        self.assertEqual('dummyValue3', results[3])

    def test_backend_multi_set_should_update_existing(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        random_key2 = uuidutils.generate_uuid(dashed=False)
        random_key3 = uuidutils.generate_uuid(dashed=False)
        mapping = {random_key1: 'dummyValue1', random_key2: 'dummyValue2', random_key3: 'dummyValue3'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertEqual('dummyValue1', region.get(random_key1))
        self.assertEqual('dummyValue2', region.get(random_key2))
        self.assertEqual('dummyValue3', region.get(random_key3))
        mapping = {random_key1: 'dummyValue4', random_key2: 'dummyValue5'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertEqual('dummyValue4', region.get(random_key1))
        self.assertEqual('dummyValue5', region.get(random_key2))
        self.assertEqual('dummyValue3', region.get(random_key3))

    def test_backend_multi_set_get_with_blanks_none(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        random_key2 = uuidutils.generate_uuid(dashed=False)
        random_key3 = uuidutils.generate_uuid(dashed=False)
        random_key4 = uuidutils.generate_uuid(dashed=False)
        mapping = {random_key1: 'dummyValue1', random_key2: None, random_key3: '', random_key4: 'dummyValue4'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertEqual('dummyValue1', region.get(random_key1))
        self.assertIsNone(region.get(random_key2))
        self.assertEqual('', region.get(random_key3))
        self.assertEqual('dummyValue4', region.get(random_key4))
        keys = [random_key, random_key1, random_key2, random_key3, random_key4]
        results = region.get_multi(keys)
        self.assertEqual(NO_VALUE, results[0])
        self.assertEqual('dummyValue1', results[1])
        self.assertIsNone(results[2])
        self.assertEqual('', results[3])
        self.assertEqual('dummyValue4', results[4])
        mapping = {random_key1: 'dummyValue5', random_key2: 'dummyValue6'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertEqual('dummyValue5', region.get(random_key1))
        self.assertEqual('dummyValue6', region.get(random_key2))
        self.assertEqual('', region.get(random_key3))

    def test_backend_delete_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue')
        self.assertEqual('dummyValue', region.get(random_key))
        region.delete(random_key)
        self.assertEqual(NO_VALUE, region.get(random_key))

    def test_backend_multi_delete_data(self):
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        random_key = uuidutils.generate_uuid(dashed=False)
        random_key1 = uuidutils.generate_uuid(dashed=False)
        random_key2 = uuidutils.generate_uuid(dashed=False)
        random_key3 = uuidutils.generate_uuid(dashed=False)
        mapping = {random_key1: 'dummyValue1', random_key2: 'dummyValue2', random_key3: 'dummyValue3'}
        region.set_multi(mapping)
        self.assertEqual(NO_VALUE, region.get(random_key))
        self.assertEqual('dummyValue1', region.get(random_key1))
        self.assertEqual('dummyValue2', region.get(random_key2))
        self.assertEqual('dummyValue3', region.get(random_key3))
        self.assertEqual(NO_VALUE, region.get('InvalidKey'))
        keys = mapping.keys()
        region.delete_multi(keys)
        self.assertEqual(NO_VALUE, region.get('InvalidKey'))
        self.assertEqual(NO_VALUE, region.get(random_key1))
        self.assertEqual(NO_VALUE, region.get(random_key2))
        self.assertEqual(NO_VALUE, region.get(random_key3))

    def test_additional_crud_method_arguments_support(self):
        """Additional arguments should works across find/insert/update."""
        self.arguments['wtimeout'] = 30000
        self.arguments['j'] = True
        self.arguments['continue_on_error'] = True
        self.arguments['secondary_acceptable_latency_ms'] = 60
        region = dp_region.make_region().configure('oslo_cache.mongo', arguments=self.arguments)
        api_methargs = region.backend.api.meth_kwargs
        self.assertEqual(30000, api_methargs['wtimeout'])
        self.assertEqual(True, api_methargs['j'])
        self.assertEqual(True, api_methargs['continue_on_error'])
        self.assertEqual(60, api_methargs['secondary_acceptable_latency_ms'])
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue1')
        self.assertEqual('dummyValue1', region.get(random_key))
        region.set(random_key, 'dummyValue2')
        self.assertEqual('dummyValue2', region.get(random_key))
        random_key = uuidutils.generate_uuid(dashed=False)
        region.set(random_key, 'dummyValue3')
        self.assertEqual('dummyValue3', region.get(random_key))