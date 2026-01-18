from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class ComponentizedTests(unittest.SynchronousTestCase, RegistryUsingMixin):
    """
    Simple test case for caching in Componentized.
    """

    def setUp(self):
        RegistryUsingMixin.setUp(self)
        components.registerAdapter(Test, AComp, ITest)
        components.registerAdapter(Test, AComp, ITest3)
        components.registerAdapter(Test2, AComp, ITest2)

    def testComponentized(self):
        components.registerAdapter(Adept, Compo, IAdept)
        components.registerAdapter(Elapsed, Compo, IElapsed)
        c = Compo()
        assert c.getComponent(IAdept).adaptorFunc() == (1, 1)
        assert c.getComponent(IAdept).adaptorFunc() == (2, 2)
        assert IElapsed(IAdept(c)).elapsedFunc() == 1

    def testInheritanceAdaptation(self):
        c = CComp()
        co1 = c.getComponent(ITest)
        co2 = c.getComponent(ITest)
        co3 = c.getComponent(ITest2)
        co4 = c.getComponent(ITest2)
        assert co1 is co2
        assert co3 is not co4
        c.removeComponent(co1)
        co5 = c.getComponent(ITest)
        co6 = c.getComponent(ITest)
        assert co5 is co6
        assert co1 is not co5

    def testMultiAdapter(self):
        c = CComp()
        co1 = c.getComponent(ITest)
        co3 = c.getComponent(ITest3)
        co4 = c.getComponent(ITest4)
        self.assertIsNone(co4)
        self.assertIs(co1, co3)

    def test_getComponentDefaults(self):
        """
        Test that a default value specified to Componentized.getComponent if
        there is no component for the requested interface.
        """
        componentized = components.Componentized()
        default = object()
        self.assertIs(componentized.getComponent(ITest, default), default)
        self.assertIs(componentized.getComponent(ITest, default=default), default)
        self.assertIs(componentized.getComponent(ITest), None)

    def test_setAdapter(self):
        """
        C{Componentized.setAdapter} sets a component for an interface by
        wrapping the instance with the given adapter class.
        """
        componentized = components.Componentized()
        componentized.setAdapter(IAdept, Adept)
        component = componentized.getComponent(IAdept)
        self.assertEqual(component.original, componentized)
        self.assertIsInstance(component, Adept)

    def test_addAdapter(self):
        """
        C{Componentized.setAdapter} adapts the instance by wrapping it with
        given adapter class, then stores it using C{addComponent}.
        """
        componentized = components.Componentized()
        componentized.addAdapter(Adept, ignoreClass=True)
        component = componentized.getComponent(IAdept)
        self.assertEqual(component.original, componentized)
        self.assertIsInstance(component, Adept)

    def test_setComponent(self):
        """
        C{Componentized.setComponent} stores the given component using the
        given interface as the key.
        """
        componentized = components.Componentized()
        obj = object()
        componentized.setComponent(ITest, obj)
        self.assertIs(componentized.getComponent(ITest), obj)

    def test_unsetComponent(self):
        """
        C{Componentized.setComponent} removes the cached component for the
        given interface.
        """
        componentized = components.Componentized()
        obj = object()
        componentized.setComponent(ITest, obj)
        componentized.unsetComponent(ITest)
        self.assertIsNone(componentized.getComponent(ITest))

    def test_reprableComponentized(self):
        """
        C{ReprableComponentized} has a C{__repr__} that lists its cache.
        """
        rc = components.ReprableComponentized()
        rc.setComponent(ITest, 'hello')
        result = repr(rc)
        self.assertIn('ITest', result)
        self.assertIn('hello', result)