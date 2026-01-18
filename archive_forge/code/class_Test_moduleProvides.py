import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_moduleProvides(unittest.TestCase):

    def test_called_from_function(self):
        from zope.interface.declarations import moduleProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
        locs = {}
        CODE = '\n'.join(['def foo():', '    moduleProvides(IFoo)'])
        exec(CODE, globs, locs)
        foo = locs['foo']
        self.assertRaises(TypeError, foo)

    def test_called_from_class(self):
        from zope.interface.declarations import moduleProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
        locs = {}
        CODE = '\n'.join(['class Foo(object):', '    moduleProvides(IFoo)'])
        with self.assertRaises(TypeError):
            exec(CODE, globs, locs)

    def test_called_once_from_module_scope(self):
        from zope.interface.declarations import moduleProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
        CODE = '\n'.join(['moduleProvides(IFoo)'])
        exec(CODE, globs)
        spec = globs['__provides__']
        self.assertEqual(list(spec), [IFoo])

    def test_called_twice_from_module_scope(self):
        from zope.interface.declarations import moduleProvides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
        CODE = '\n'.join(['moduleProvides(IFoo)', 'moduleProvides(IFoo)'])
        with self.assertRaises(TypeError):
            exec(CODE, globs)