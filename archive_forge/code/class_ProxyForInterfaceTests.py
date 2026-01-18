from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class ProxyForInterfaceTests(unittest.SynchronousTestCase):
    """
    Tests for L{proxyForInterface}.
    """

    def test_original(self):
        """
        Proxy objects should have an C{original} attribute which refers to the
        original object passed to the constructor.
        """
        original = object()
        proxy = proxyForInterface(IProxiedInterface)(original)
        self.assertIs(proxy.original, original)

    def test_proxyMethod(self):
        """
        The class created from L{proxyForInterface} passes methods on an
        interface to the object which is passed to its constructor.
        """
        klass = proxyForInterface(IProxiedInterface)
        yayable = Yayable()
        proxy = klass(yayable)
        proxy.yay()
        self.assertEqual(proxy.yay(), 2)
        self.assertEqual(yayable.yays, 2)

    def test_decoratedProxyMethod(self):
        """
        Methods of the class created from L{proxyForInterface} can be used with
        the decorator-helper L{functools.wraps}.
        """
        base = proxyForInterface(IProxiedInterface)

        class klass(base):

            @wraps(base.yay)
            def yay(self):
                self.original.yays += 1
                return base.yay(self)
        original = Yayable()
        yayable = klass(original)
        yayable.yay()
        self.assertEqual(2, original.yays)

    def test_proxyAttribute(self):
        """
        Proxy objects should proxy declared attributes, but not other
        attributes.
        """
        yayable = Yayable()
        yayable.ifaceAttribute = object()
        proxy = proxyForInterface(IProxiedInterface)(yayable)
        self.assertIs(proxy.ifaceAttribute, yayable.ifaceAttribute)
        self.assertRaises(AttributeError, lambda: proxy.yays)

    def test_proxySetAttribute(self):
        """
        The attributes that proxy objects proxy should be assignable and affect
        the original object.
        """
        yayable = Yayable()
        proxy = proxyForInterface(IProxiedInterface)(yayable)
        thingy = object()
        proxy.ifaceAttribute = thingy
        self.assertIs(yayable.ifaceAttribute, thingy)

    def test_proxyDeleteAttribute(self):
        """
        The attributes that proxy objects proxy should be deletable and affect
        the original object.
        """
        yayable = Yayable()
        yayable.ifaceAttribute = None
        proxy = proxyForInterface(IProxiedInterface)(yayable)
        del proxy.ifaceAttribute
        self.assertFalse(hasattr(yayable, 'ifaceAttribute'))

    def test_multipleMethods(self):
        """
        [Regression test] The proxy should send its method calls to the correct
        method, not the incorrect one.
        """
        multi = MultipleMethodImplementor()
        proxy = proxyForInterface(IMultipleMethods)(multi)
        self.assertEqual(proxy.methodOne(), 1)
        self.assertEqual(proxy.methodTwo(), 2)

    def test_subclassing(self):
        """
        It is possible to subclass the result of L{proxyForInterface}.
        """

        class SpecializedProxy(proxyForInterface(IProxiedInterface)):
            """
            A specialized proxy which can decrement the number of yays.
            """

            def boo(self):
                """
                Decrement the number of yays.
                """
                self.original.yays -= 1
        yayable = Yayable()
        special = SpecializedProxy(yayable)
        self.assertEqual(yayable.yays, 0)
        special.boo()
        self.assertEqual(yayable.yays, -1)

    def test_proxyName(self):
        """
        The name of a proxy class indicates which interface it proxies.
        """
        proxy = proxyForInterface(IProxiedInterface)
        self.assertEqual(proxy.__name__, '(Proxy for twisted.python.test.test_components.IProxiedInterface)')

    def test_implements(self):
        """
        The resulting proxy implements the interface that it proxies.
        """
        proxy = proxyForInterface(IProxiedInterface)
        self.assertTrue(IProxiedInterface.implementedBy(proxy))

    def test_proxyDescriptorGet(self):
        """
        _ProxyDescriptor's __get__ method should return the appropriate
        attribute of its argument's 'original' attribute if it is invoked with
        an object.  If it is invoked with None, it should return a false
        class-method emulator instead.

        For some reason, Python's documentation recommends to define
        descriptors' __get__ methods with the 'type' parameter as optional,
        despite the fact that Python itself never actually calls the descriptor
        that way.  This is probably do to support 'foo.__get__(bar)' as an
        idiom.  Let's make sure that the behavior is correct.  Since we don't
        actually use the 'type' argument at all, this test calls it the
        idiomatic way to ensure that signature works; test_proxyInheritance
        verifies the how-Python-actually-calls-it signature.
        """

        class Sample:
            called = False

            def hello(self):
                self.called = True
        fakeProxy = Sample()
        testObject = Sample()
        fakeProxy.original = testObject
        pd = components._ProxyDescriptor('hello', 'original')
        self.assertEqual(pd.__get__(fakeProxy), testObject.hello)
        fakeClassMethod = pd.__get__(None)
        fakeClassMethod(fakeProxy)
        self.assertTrue(testObject.called)

    def test_proxyInheritance(self):
        """
        Subclasses of the class returned from L{proxyForInterface} should be
        able to upcall methods by reference to their superclass, as any normal
        Python class can.
        """

        class YayableWrapper(proxyForInterface(IProxiedInterface)):
            """
            This class does not override any functionality.
            """

        class EnhancedWrapper(YayableWrapper):
            """
            This class overrides the 'yay' method.
            """
            wrappedYays = 1

            def yay(self, *a, **k):
                self.wrappedYays += 1
                return YayableWrapper.yay(self, *a, **k) + 7
        yayable = Yayable()
        wrapper = EnhancedWrapper(yayable)
        self.assertEqual(wrapper.yay(3, 4, x=5, y=6), 8)
        self.assertEqual(yayable.yayArgs, [((3, 4), dict(x=5, y=6))])

    def test_interfaceInheritance(self):
        """
        Proxies of subinterfaces generated with proxyForInterface should allow
        access to attributes of both the child and the base interfaces.
        """
        proxyClass = proxyForInterface(IProxiedSubInterface)
        booable = Booable()
        proxy = proxyClass(booable)
        proxy.yay()
        proxy.boo()
        self.assertTrue(booable.yayed)
        self.assertTrue(booable.booed)

    def test_attributeCustomization(self):
        """
        The original attribute name can be customized via the
        C{originalAttribute} argument of L{proxyForInterface}: the attribute
        should change, but the methods of the original object should still be
        callable, and the attributes still accessible.
        """
        yayable = Yayable()
        yayable.ifaceAttribute = object()
        proxy = proxyForInterface(IProxiedInterface, originalAttribute='foo')(yayable)
        self.assertIs(proxy.foo, yayable)
        self.assertEqual(proxy.yay(), 1)
        self.assertIs(proxy.ifaceAttribute, yayable.ifaceAttribute)
        thingy = object()
        proxy.ifaceAttribute = thingy
        self.assertIs(yayable.ifaceAttribute, thingy)
        del proxy.ifaceAttribute
        self.assertFalse(hasattr(yayable, 'ifaceAttribute'))