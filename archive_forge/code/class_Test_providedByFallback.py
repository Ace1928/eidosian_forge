import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
class Test_providedByFallback(unittest.TestCase):

    def _getFallbackClass(self):
        from zope.interface.declarations import providedByFallback
        return providedByFallback
    _getTargetClass = _getFallbackClass

    def _callFUT(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_wo_providedBy_on_class_wo_implements(self):

        class Foo:
            pass
        foo = Foo()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [])

    def test_w_providedBy_valid_spec(self):
        from zope.interface.declarations import Provides
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = Provides(Foo, IFoo)
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [IFoo])

    def test_w_providedBy_invalid_spec(self):

        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = object()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [])

    def test_w_providedBy_invalid_spec_class_w_implements(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = object()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [IFoo])

    def test_w_providedBy_invalid_spec_w_provides_no_provides_on_class(self):

        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = object()
        expected = foo.__provides__ = object()
        spec = self._callFUT(foo)
        self.assertTrue(spec is expected)

    def test_w_providedBy_invalid_spec_w_provides_diff_provides_on_class(self):

        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = object()
        expected = foo.__provides__ = object()
        Foo.__provides__ = object()
        spec = self._callFUT(foo)
        self.assertTrue(spec is expected)

    def test_w_providedBy_invalid_spec_w_provides_same_provides_on_class(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        foo.__providedBy__ = object()
        foo.__provides__ = Foo.__provides__ = object()
        spec = self._callFUT(foo)
        self.assertEqual(list(spec), [IFoo])

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
        derived = Derived()
        self.assertEqual(list(self._callFUT(derived)), [IDerived, IBase])
        sup = super(Derived, derived)
        fut = self._callFUT(sup)
        self.assertIsNone(fut._dependents)
        self.assertEqual(list(fut), [IBase])

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
        derived = Derived()
        self.assertEqual(list(self._callFUT(derived)), [IDerived])
        sup = super(Derived, derived)
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
        derived = Derived()
        self.assertEqual(list(self._callFUT(derived)), [IDerived])
        sup = super(Derived, derived)
        fut = self._callFUT(sup)
        self.assertIsNone(fut._dependents)
        self.assertEqual(list(fut), [])

    def test_super_when_object_directly_provides(self):
        from zope.interface import Interface
        from zope.interface.declarations import directlyProvides
        from zope.interface.declarations import implementer

        class IBase(Interface):
            pass

        class IDerived(IBase):
            pass

        @implementer(IBase)
        class Base:
            pass

        class Derived(Base):
            pass
        derived = Derived()
        self.assertEqual(list(self._callFUT(derived)), [IBase])
        directlyProvides(derived, IDerived)
        self.assertEqual(list(self._callFUT(derived)), [IDerived, IBase])
        sup = super(Derived, derived)
        fut = self._callFUT(sup)
        self.assertIsNone(fut._dependents)
        self.assertEqual(list(fut), [IBase])

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
        d = Derived()
        sd = super(Derived, d)
        sm1 = super(M1, d)
        sm2 = super(M2, d)
        self.assertEqual(list(self._callFUT(d)), [IDerived, IUnrelated, IM1, IBase, IM2])
        self.assertEqual(list(self._callFUT(sd)), [IM1, IBase, IM2])
        self.assertEqual(list(self._callFUT(sm1)), [IM2, IBase])
        self.assertEqual(list(self._callFUT(sm2)), [IBase])

    def test_catches_only_AttributeError_on_providedBy(self):
        MissingSomeAttrs.test_raises(self, self._callFUT, expected_missing='__providedBy__', __class__=object)

    def test_catches_only_AttributeError_on_class(self):
        MissingSomeAttrs.test_raises(self, self._callFUT, expected_missing='__class__')