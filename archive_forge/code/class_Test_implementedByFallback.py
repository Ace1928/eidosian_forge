import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_implementedByFallback(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.declarations import implementedByFallback
        return implementedByFallback
    _getFallbackClass = _getTargetClass

    def _callFUT(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_dictless_wo_existing_Implements_wo_registrations(self):

        class Foo:
            __slots__ = ('__implemented__',)
        foo = Foo()
        foo.__implemented__ = None
        self.assertEqual(list(self._callFUT(foo)), [])

    def test_dictless_wo_existing_Implements_cant_assign___implemented__(self):

        class Foo:

            def _get_impl(self):
                raise NotImplementedError()

            def _set_impl(self, val):
                raise TypeError
            __implemented__ = property(_get_impl, _set_impl)

            def __call__(self):
                raise NotImplementedError()
        foo = Foo()
        self.assertRaises(TypeError, self._callFUT, foo)

    def test_dictless_wo_existing_Implements_w_registrations(self):
        from zope.interface import declarations

        class Foo:
            __slots__ = ('__implemented__',)
        foo = Foo()
        foo.__implemented__ = None
        reg = object()
        with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
            specs[foo] = reg
            self.assertTrue(self._callFUT(foo) is reg)

    def test_dictless_w_existing_Implements(self):
        from zope.interface.declarations import Implements
        impl = Implements()

        class Foo:
            __slots__ = ('__implemented__',)
        foo = Foo()
        foo.__implemented__ = impl
        self.assertTrue(self._callFUT(foo) is impl)

    def test_dictless_w_existing_not_Implements(self):
        from zope.interface.interface import InterfaceClass

        class Foo:
            __slots__ = ('__implemented__',)
        foo = Foo()
        IFoo = InterfaceClass('IFoo')
        foo.__implemented__ = (IFoo,)
        self.assertEqual(list(self._callFUT(foo)), [IFoo])

    def test_w_existing_attr_as_Implements(self):
        from zope.interface.declarations import Implements
        impl = Implements()

        class Foo:
            __implemented__ = impl
        self.assertTrue(self._callFUT(Foo) is impl)

    def test_builtins_added_to_cache(self):
        from zope.interface import declarations
        from zope.interface.declarations import Implements
        with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
            self.assertEqual(list(self._callFUT(tuple)), [])
            self.assertEqual(list(self._callFUT(list)), [])
            self.assertEqual(list(self._callFUT(dict)), [])
            for typ in (tuple, list, dict):
                spec = specs[typ]
                self.assertIsInstance(spec, Implements)
                self.assertEqual(repr(spec), 'classImplements(%s)' % (typ.__name__,))

    def test_builtins_w_existing_cache(self):
        from zope.interface import declarations
        t_spec, l_spec, d_spec = (object(), object(), object())
        with _MonkeyDict(declarations, 'BuiltinImplementationSpecifications') as specs:
            specs[tuple] = t_spec
            specs[list] = l_spec
            specs[dict] = d_spec
            self.assertTrue(self._callFUT(tuple) is t_spec)
            self.assertTrue(self._callFUT(list) is l_spec)
            self.assertTrue(self._callFUT(dict) is d_spec)

    def test_oldstyle_class_no_assertions(self):

        class Foo:
            pass
        self.assertEqual(list(self._callFUT(Foo)), [])

    def test_no_assertions(self):

        class Foo:
            pass
        self.assertEqual(list(self._callFUT(Foo)), [])

    def test_w_None_no_bases_not_factory(self):

        class Foo:
            __implemented__ = None
        foo = Foo()
        self.assertRaises(TypeError, self._callFUT, foo)

    def test_w_None_no_bases_w_factory(self):
        from zope.interface.declarations import objectSpecificationDescriptor

        class Foo:
            __implemented__ = None

            def __call__(self):
                raise NotImplementedError()
        foo = Foo()
        foo.__name__ = 'foo'
        spec = self._callFUT(foo)
        self.assertEqual(spec.__name__, 'zope.interface.tests.test_declarations.foo')
        self.assertIs(spec.inherit, foo)
        self.assertIs(foo.__implemented__, spec)
        self.assertIs(foo.__providedBy__, objectSpecificationDescriptor)
        self.assertNotIn('__provides__', foo.__dict__)

    def test_w_None_no_bases_w_class(self):
        from zope.interface.declarations import ClassProvides

        class Foo:
            __implemented__ = None
        spec = self._callFUT(Foo)
        self.assertEqual(spec.__name__, 'zope.interface.tests.test_declarations.Foo')
        self.assertIs(spec.inherit, Foo)
        self.assertIs(Foo.__implemented__, spec)
        self.assertIsInstance(Foo.__providedBy__, ClassProvides)
        self.assertIsInstance(Foo.__provides__, ClassProvides)
        self.assertEqual(Foo.__provides__, Foo.__providedBy__)

    def test_w_existing_Implements(self):
        from zope.interface.declarations import Implements
        impl = Implements()

        class Foo:
            __implemented__ = impl
        self.assertTrue(self._callFUT(Foo) is impl)

    def test_super_when_base_implements_interface(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        @implementer(IBase)
        class Base:
            pass

        @implementer(IDerived)
        class Derived(Base):
            pass
        self.assertEqual(list(self._callFUT(Derived)), [IDerived, IBase])
        sup = super(Derived, Derived)
        self.assertEqual(list(self._callFUT(sup)), [IBase])

    def test_super_when_base_implements_interface_diamond(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        @implementer(IBase)
        class Base:
            pass

        class Child1(Base):
            pass

        class Child2(Base):
            pass

        @implementer(IDerived)
        class Derived(Child1, Child2):
            pass
        self.assertEqual(list(self._callFUT(Derived)), [IDerived, IBase])
        sup = super(Derived, Derived)
        self.assertEqual(list(self._callFUT(sup)), [IBase])

    def test_super_when_parent_implements_interface_diamond(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        class Base:
            pass

        class Child1(Base):
            pass

        @implementer(IBase)
        class Child2(Base):
            pass

        @implementer(IDerived)
        class Derived(Child1, Child2):
            pass
        self.assertEqual(Derived.__mro__, (Derived, Child1, Child2, Base, object))
        self.assertEqual(list(self._callFUT(Derived)), [IDerived, IBase])
        sup = super(Derived, Derived)
        fut = self._callFUT(sup)
        self.assertEqual(list(fut), [IBase])
        self.assertIsNone(fut._dependents)

    def test_super_when_base_doesnt_implement_interface(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        class Base:
            pass

        @implementer(IDerived)
        class Derived(Base):
            pass
        self.assertEqual(list(self._callFUT(Derived)), [IDerived])
        sup = super(Derived, Derived)
        self.assertEqual(list(self._callFUT(sup)), [])

    def test_super_when_base_is_object(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        @implementer(IDerived)
        class Derived:
            pass
        self.assertEqual(list(self._callFUT(Derived)), [IDerived])
        sup = super(Derived, Derived)
        self.assertEqual(list(self._callFUT(sup)), [])

    def test_super_multi_level_multi_inheritance(self):
        from zope.interface import Interface
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IM1(Interface):
            pass

        class IM2(Interface):
            pass

        class IDerived(IBase):
            pass

        class IUnrelated(Interface):
            pass

        @implementer(IBase)
        class Base:
            pass

        @implementer(IM1)
        class M1(Base):
            pass

        @implementer(IM2)
        class M2(Base):
            pass

        @implementer(IDerived, IUnrelated)
        class Derived(M1, M2):
            pass
        d = Derived
        sd = super(Derived, Derived)
        sm1 = super(M1, Derived)
        sm2 = super(M2, Derived)
        self.assertEqual(list(self._callFUT(d)), [IDerived, IUnrelated, IM1, IBase, IM2])
        self.assertEqual(list(self._callFUT(sd)), [IM1, IBase, IM2])
        self.assertEqual(list(self._callFUT(sm1)), [IM2, IBase])
        self.assertEqual(list(self._callFUT(sm2)), [IBase])