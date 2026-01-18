import os
import sys
from breezy import branch, osutils, registry, tests
class TestRegistryWithDirs(tests.TestCaseInTempDir):
    """Registry tests that require temporary dirs"""

    def create_plugin_file(self, contents):
        """Create a file to be used as a plugin.

        This is created in a temporary directory, so that we
        are sure that it doesn't start in the plugin path.
        """
        os.mkdir('tmp')
        plugin_name = 'bzr_plugin_a_{}'.format(osutils.rand_chars(4))
        with open('tmp/' + plugin_name + '.py', 'wb') as f:
            f.write(contents)
        return plugin_name

    def create_simple_plugin(self):
        return self.create_plugin_file(b'object1 = "foo"\n\n\ndef function(a,b,c):\n    return a,b,c\n\n\nclass MyClass(object):\n    def __init__(self, a):\n      self.a = a\n\n\n')

    def test_lazy_import_registry_foo(self):
        a_registry = registry.Registry()
        a_registry.register_lazy('foo', 'breezy.branch', 'Branch')
        a_registry.register_lazy('bar', 'breezy.branch', 'Branch.hooks')
        self.assertEqual(branch.Branch, a_registry.get('foo'))
        self.assertEqual(branch.Branch.hooks, a_registry.get('bar'))

    def test_lazy_import_registry(self):
        plugin_name = self.create_simple_plugin()
        a_registry = registry.Registry()
        a_registry.register_lazy('obj', plugin_name, 'object1')
        a_registry.register_lazy('function', plugin_name, 'function')
        a_registry.register_lazy('klass', plugin_name, 'MyClass')
        a_registry.register_lazy('module', plugin_name, None)
        self.assertEqual(['function', 'klass', 'module', 'obj'], sorted(a_registry.keys()))
        self.assertFalse(plugin_name in sys.modules)
        self.assertRaises(ImportError, a_registry.get, 'obj')
        plugin_path = self.test_dir + '/tmp'
        sys.path.append(plugin_path)
        try:
            obj = a_registry.get('obj')
            self.assertEqual('foo', obj)
            self.assertTrue(plugin_name in sys.modules)
            func = a_registry.get('function')
            self.assertEqual(plugin_name, func.__module__)
            self.assertEqual('function', func.__name__)
            self.assertEqual((1, [], '3'), func(1, [], '3'))
            klass = a_registry.get('klass')
            self.assertEqual(plugin_name, klass.__module__)
            self.assertEqual('MyClass', klass.__name__)
            inst = klass(1)
            self.assertIsInstance(inst, klass)
            self.assertEqual(1, inst.a)
            module = a_registry.get('module')
            self.assertIs(obj, module.object1)
            self.assertIs(func, module.function)
            self.assertIs(klass, module.MyClass)
        finally:
            sys.path.remove(plugin_path)

    def test_lazy_import_get_module(self):
        a_registry = registry.Registry()
        a_registry.register_lazy('obj', 'breezy.tests.test_registry', 'object1')
        self.assertEqual('breezy.tests.test_registry', a_registry._get_module('obj'))

    def test_normal_get_module(self):

        class AThing:
            """Something"""
        a_registry = registry.Registry()
        a_registry.register('obj', AThing())
        self.assertEqual('breezy.tests.test_registry', a_registry._get_module('obj'))