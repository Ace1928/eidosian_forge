import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
class GetPackageForModuleTest(test_util.TestCase):

    def setUp(self):
        self.original_modules = dict(sys.modules)

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.original_modules)

    def CreateModule(self, name, file_name=None):
        if file_name is None:
            file_name = '%s.py' % name
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    def assertPackageEquals(self, expected, actual):
        self.assertEquals(expected, actual)
        if actual is not None:
            self.assertTrue(isinstance(actual, six.text_type))

    def testByString(self):
        module = self.CreateModule('service_module')
        module.package = 'my_package'
        self.assertPackageEquals('my_package', util.get_package_for_module('service_module'))

    def testModuleNameNotInSys(self):
        self.assertPackageEquals(None, util.get_package_for_module('service_module'))

    def testHasPackage(self):
        module = self.CreateModule('service_module')
        module.package = 'my_package'
        self.assertPackageEquals('my_package', util.get_package_for_module(module))

    def testHasModuleName(self):
        module = self.CreateModule('service_module')
        self.assertPackageEquals('service_module', util.get_package_for_module(module))

    def testIsMain(self):
        module = self.CreateModule('__main__')
        module.__file__ = '/bing/blam/bloom/blarm/my_file.py'
        self.assertPackageEquals('my_file', util.get_package_for_module(module))

    def testIsMainCompiled(self):
        module = self.CreateModule('__main__')
        module.__file__ = '/bing/blam/bloom/blarm/my_file.pyc'
        self.assertPackageEquals('my_file', util.get_package_for_module(module))

    def testNoExtension(self):
        module = self.CreateModule('__main__')
        module.__file__ = '/bing/blam/bloom/blarm/my_file'
        self.assertPackageEquals('my_file', util.get_package_for_module(module))

    def testNoPackageAtAll(self):
        module = self.CreateModule('__main__')
        self.assertPackageEquals('__main__', util.get_package_for_module(module))