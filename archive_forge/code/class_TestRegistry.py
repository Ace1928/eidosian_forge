import os
import sys
from breezy import branch, osutils, registry, tests
class TestRegistry(tests.TestCase):

    def register_stuff(self, a_registry):
        a_registry.register('one', 1)
        a_registry.register('two', 2)
        a_registry.register('four', 4)
        a_registry.register('five', 5)

    def test_registry(self):
        a_registry = registry.Registry()
        self.register_stuff(a_registry)
        self.assertTrue(a_registry.default_key is None)
        self.assertRaises(KeyError, a_registry.get)
        self.assertRaises(KeyError, a_registry.get, None)
        self.assertEqual(2, a_registry.get('two'))
        self.assertRaises(KeyError, a_registry.get, 'three')
        a_registry.default_key = 'five'
        self.assertTrue(a_registry.default_key == 'five')
        self.assertEqual(5, a_registry.get())
        self.assertEqual(5, a_registry.get(None))
        self.assertRaises(KeyError, a_registry.get, 'six')
        self.assertRaises(KeyError, a_registry._set_default_key, 'six')
        self.assertEqual(['five', 'four', 'one', 'two'], a_registry.keys())

    def test_registry_funcs(self):
        a_registry = registry.Registry()
        self.register_stuff(a_registry)
        self.assertTrue('one' in a_registry)
        a_registry.remove('one')
        self.assertFalse('one' in a_registry)
        self.assertRaises(KeyError, a_registry.get, 'one')
        a_registry.register('one', 'one')
        self.assertEqual(['five', 'four', 'one', 'two'], sorted(a_registry.keys()))
        self.assertEqual([('five', 5), ('four', 4), ('one', 'one'), ('two', 2)], sorted(a_registry.iteritems()))

    def test_register_override(self):
        a_registry = registry.Registry()
        a_registry.register('one', 'one')
        self.assertRaises(KeyError, a_registry.register, 'one', 'two')
        self.assertRaises(KeyError, a_registry.register, 'one', 'two', override_existing=False)
        a_registry.register('one', 'two', override_existing=True)
        self.assertEqual('two', a_registry.get('one'))
        self.assertRaises(KeyError, a_registry.register_lazy, 'one', 'three', 'four')
        a_registry.register_lazy('one', 'module', 'member', override_existing=True)

    def test_registry_help(self):
        a_registry = registry.Registry()
        a_registry.register('one', 1, help='help text for one')
        a_registry.register_lazy('two', 'nonexistent_module', 'member', help='help text for two')
        help_calls = []

        def generic_help(reg, key):
            help_calls.append(key)
            return 'generic help for {}'.format(key)
        a_registry.register('three', 3, help=generic_help)
        a_registry.register_lazy('four', 'nonexistent_module', 'member2', help=generic_help)
        a_registry.register('five', 5)

        def help_from_object(reg, key):
            obj = reg.get(key)
            return obj.help()

        class SimpleObj:

            def help(self):
                return 'this is my help'
        a_registry.register('six', SimpleObj(), help=help_from_object)
        self.assertEqual('help text for one', a_registry.get_help('one'))
        self.assertEqual('help text for two', a_registry.get_help('two'))
        self.assertEqual('generic help for three', a_registry.get_help('three'))
        self.assertEqual(['three'], help_calls)
        self.assertEqual('generic help for four', a_registry.get_help('four'))
        self.assertEqual(['three', 'four'], help_calls)
        self.assertEqual(None, a_registry.get_help('five'))
        self.assertEqual('this is my help', a_registry.get_help('six'))
        self.assertRaises(KeyError, a_registry.get_help, None)
        self.assertRaises(KeyError, a_registry.get_help, 'seven')
        a_registry.default_key = 'one'
        self.assertEqual('help text for one', a_registry.get_help(None))
        self.assertRaises(KeyError, a_registry.get_help, 'seven')
        self.assertEqual([('five', None), ('four', 'generic help for four'), ('one', 'help text for one'), ('six', 'this is my help'), ('three', 'generic help for three'), ('two', 'help text for two')], sorted(((key, a_registry.get_help(key)) for key in a_registry.keys())))
        self.assertEqual(['four', 'four', 'three', 'three'], sorted(help_calls))

    def test_registry_info(self):
        a_registry = registry.Registry()
        a_registry.register('one', 1, info='string info')
        a_registry.register_lazy('two', 'nonexistent_module', 'member', info=2)
        a_registry.register('three', 3, info=['a', 'list'])
        obj = object()
        a_registry.register_lazy('four', 'nonexistent_module', 'member2', info=obj)
        a_registry.register('five', 5)
        self.assertEqual('string info', a_registry.get_info('one'))
        self.assertEqual(2, a_registry.get_info('two'))
        self.assertEqual(['a', 'list'], a_registry.get_info('three'))
        self.assertIs(obj, a_registry.get_info('four'))
        self.assertIs(None, a_registry.get_info('five'))
        self.assertRaises(KeyError, a_registry.get_info, None)
        self.assertRaises(KeyError, a_registry.get_info, 'six')
        a_registry.default_key = 'one'
        self.assertEqual('string info', a_registry.get_info(None))
        self.assertRaises(KeyError, a_registry.get_info, 'six')
        self.assertEqual([('five', None), ('four', obj), ('one', 'string info'), ('three', ['a', 'list']), ('two', 2)], sorted(((key, a_registry.get_info(key)) for key in a_registry.keys())))

    def test_get_prefix(self):
        my_registry = registry.Registry()
        http_object = object()
        sftp_object = object()
        my_registry.register('http:', http_object)
        my_registry.register('sftp:', sftp_object)
        found_object, suffix = my_registry.get_prefix('http://foo/bar')
        self.assertEqual('//foo/bar', suffix)
        self.assertIs(http_object, found_object)
        self.assertIsNot(sftp_object, found_object)
        found_object, suffix = my_registry.get_prefix('sftp://baz/qux')
        self.assertEqual('//baz/qux', suffix)
        self.assertIs(sftp_object, found_object)

    def test_registry_alias(self):
        a_registry = registry.Registry()
        a_registry.register('one', 1, info='string info')
        a_registry.register_alias('two', 'one')
        a_registry.register_alias('three', 'one', info='own info')
        self.assertEqual(a_registry.get('one'), a_registry.get('two'))
        self.assertEqual(a_registry.get_help('one'), a_registry.get_help('two'))
        self.assertEqual(a_registry.get_info('one'), a_registry.get_info('two'))
        self.assertEqual('own info', a_registry.get_info('three'))
        self.assertEqual({'two': 'one', 'three': 'one'}, a_registry.aliases())
        self.assertEqual({'one': ['three', 'two']}, {k: sorted(v) for k, v in a_registry.alias_map().items()})

    def test_registry_alias_exists(self):
        a_registry = registry.Registry()
        a_registry.register('one', 1, info='string info')
        a_registry.register('two', 2)
        self.assertRaises(KeyError, a_registry.register_alias, 'one', 'one')

    def test_registry_alias_targetmissing(self):
        a_registry = registry.Registry()
        self.assertRaises(KeyError, a_registry.register_alias, 'one', 'two')