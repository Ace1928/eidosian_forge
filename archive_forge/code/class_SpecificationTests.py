import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class SpecificationTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.interface import Specification
        return Specification

    def _makeOne(self, bases=_marker):
        if bases is _marker:
            return self._getTargetClass()()
        return self._getTargetClass()(bases)

    def test_ctor(self):
        from zope.interface.interface import Interface
        spec = self._makeOne()
        self.assertEqual(spec.__bases__, ())
        self.assertEqual(len(spec._implied), 2)
        self.assertTrue(spec in spec._implied)
        self.assertTrue(Interface in spec._implied)
        self.assertEqual(len(spec.dependents), 0)

    def test_subscribe_first_time(self):
        spec = self._makeOne()
        dep = DummyDependent()
        spec.subscribe(dep)
        self.assertEqual(len(spec.dependents), 1)
        self.assertEqual(spec.dependents[dep], 1)

    def test_subscribe_again(self):
        spec = self._makeOne()
        dep = DummyDependent()
        spec.subscribe(dep)
        spec.subscribe(dep)
        self.assertEqual(spec.dependents[dep], 2)

    def test_unsubscribe_miss(self):
        spec = self._makeOne()
        dep = DummyDependent()
        self.assertRaises(KeyError, spec.unsubscribe, dep)

    def test_unsubscribe(self):
        spec = self._makeOne()
        dep = DummyDependent()
        spec.subscribe(dep)
        spec.subscribe(dep)
        spec.unsubscribe(dep)
        self.assertEqual(spec.dependents[dep], 1)
        spec.unsubscribe(dep)
        self.assertFalse(dep in spec.dependents)

    def test___setBases_subscribes_bases_and_notifies_dependents(self):
        from zope.interface.interface import Interface
        spec = self._makeOne()
        dep = DummyDependent()
        spec.subscribe(dep)

        class I(Interface):
            pass

        class J(Interface):
            pass
        spec.__bases__ = (I,)
        self.assertEqual(dep._changed, [spec])
        self.assertEqual(I.dependents[spec], 1)
        spec.__bases__ = (J,)
        self.assertEqual(I.dependents.get(spec), None)
        self.assertEqual(J.dependents[spec], 1)

    def test_changed_clears_volatiles_and_implied(self):
        from zope.interface.interface import Interface

        class I(Interface):
            pass
        spec = self._makeOne()
        spec._v_attrs = 'Foo'
        spec._implied[I] = ()
        spec.changed(spec)
        self.assertIsNone(spec._v_attrs)
        self.assertFalse(I in spec._implied)

    def test_interfaces_skips_already_seen(self):
        from zope.interface.interface import Interface

        class IFoo(Interface):
            pass
        spec = self._makeOne([IFoo, IFoo])
        self.assertEqual(list(spec.interfaces()), [IFoo])

    def test_extends_strict_wo_self(self):
        from zope.interface.interface import Interface

        class IFoo(Interface):
            pass
        spec = self._makeOne(IFoo)
        self.assertFalse(spec.extends(IFoo, strict=True))

    def test_extends_strict_w_self(self):
        spec = self._makeOne()
        self.assertFalse(spec.extends(spec, strict=True))

    def test_extends_non_strict_w_self(self):
        spec = self._makeOne()
        self.assertTrue(spec.extends(spec, strict=False))

    def test_get_hit_w__v_attrs(self):
        spec = self._makeOne()
        foo = object()
        spec._v_attrs = {'foo': foo}
        self.assertTrue(spec.get('foo') is foo)

    def test_get_hit_from_base_wo__v_attrs(self):
        from zope.interface.interface import Attribute
        from zope.interface.interface import Interface

        class IFoo(Interface):
            foo = Attribute('foo')

        class IBar(Interface):
            bar = Attribute('bar')
        spec = self._makeOne([IFoo, IBar])
        self.assertTrue(spec.get('foo') is IFoo.get('foo'))
        self.assertTrue(spec.get('bar') is IBar.get('bar'))

    def test_multiple_inheritance_no_interfaces(self):
        from zope.interface.declarations import implementedBy
        from zope.interface.declarations import implementer
        from zope.interface.interface import Interface

        class IDefaultViewName(Interface):
            pass

        class Context:
            pass

        class RDBModel(Context):
            pass

        class IOther(Interface):
            pass

        @implementer(IOther)
        class OtherBase:
            pass

        class Model(OtherBase, Context):
            pass
        self.assertEqual(implementedBy(Model).__sro__, (implementedBy(Model), implementedBy(OtherBase), IOther, implementedBy(Context), implementedBy(object), Interface))