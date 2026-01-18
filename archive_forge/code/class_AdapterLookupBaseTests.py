import unittest
from zope.interface.tests import OptimizationTestMixin
class AdapterLookupBaseTests(unittest.TestCase):

    def _getTargetClass(self):
        from zope.interface.adapter import AdapterLookupBase
        return AdapterLookupBase

    def _makeOne(self, registry):
        return self._getTargetClass()(registry)

    def _makeSubregistry(self, *provided):

        class Subregistry:

            def __init__(self):
                self._adapters = []
                self._subscribers = []
        return Subregistry()

    def _makeRegistry(self, *provided):

        class Registry:

            def __init__(self, provided):
                self._provided = provided
                self.ro = []
        return Registry(provided)

    def test_ctor_empty_registry(self):
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        self.assertEqual(alb._extendors, {})

    def test_ctor_w_registry_provided(self):
        from zope.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        alb = self._makeOne(registry)
        self.assertEqual(sorted(alb._extendors.keys()), sorted([IBar, IFoo, Interface]))
        self.assertEqual(alb._extendors[IFoo], [IFoo, IBar])
        self.assertEqual(alb._extendors[IBar], [IBar])
        self.assertEqual(sorted(alb._extendors[Interface]), sorted([IFoo, IBar]))

    def test_changed_empty_required(self):

        class Mixin:

            def changed(self, *other):
                pass

        class Derived(self._getTargetClass(), Mixin):
            pass
        registry = self._makeRegistry()
        alb = Derived(registry)
        alb.changed(alb)

    def test_changed_w_required(self):

        class Mixin:

            def changed(self, *other):
                pass

        class Derived(self._getTargetClass(), Mixin):
            pass

        class FauxWeakref:
            _unsub = None

            def __init__(self, here):
                self._here = here

            def __call__(self):
                return self if self._here else None

            def unsubscribe(self, target):
                self._unsub = target
        gone = FauxWeakref(False)
        here = FauxWeakref(True)
        registry = self._makeRegistry()
        alb = Derived(registry)
        alb._required[gone] = 1
        alb._required[here] = 1
        alb.changed(alb)
        self.assertEqual(len(alb._required), 0)
        self.assertEqual(gone._unsub, None)
        self.assertEqual(here._unsub, alb)

    def test_init_extendors_after_registry_update(self):
        from zope.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        registry._provided = [IFoo, IBar]
        alb.init_extendors()
        self.assertEqual(sorted(alb._extendors.keys()), sorted([IBar, IFoo, Interface]))
        self.assertEqual(alb._extendors[IFoo], [IFoo, IBar])
        self.assertEqual(alb._extendors[IBar], [IBar])
        self.assertEqual(sorted(alb._extendors[Interface]), sorted([IFoo, IBar]))

    def test_add_extendor(self):
        from zope.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        alb.add_extendor(IFoo)
        alb.add_extendor(IBar)
        self.assertEqual(sorted(alb._extendors.keys()), sorted([IBar, IFoo, Interface]))
        self.assertEqual(alb._extendors[IFoo], [IFoo, IBar])
        self.assertEqual(alb._extendors[IBar], [IBar])
        self.assertEqual(sorted(alb._extendors[Interface]), sorted([IFoo, IBar]))

    def test_remove_extendor(self):
        from zope.interface import Interface
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        alb = self._makeOne(registry)
        alb.remove_extendor(IFoo)
        self.assertEqual(sorted(alb._extendors.keys()), sorted([IFoo, IBar, Interface]))
        self.assertEqual(alb._extendors[IFoo], [IBar])
        self.assertEqual(alb._extendors[IBar], [IBar])
        self.assertEqual(sorted(alb._extendors[Interface]), sorted([IBar]))

    def test__uncached_lookup_empty_ro(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertEqual(result, None)
        self.assertEqual(len(alb._required), 1)
        self.assertIn(IFoo.weakref(), alb._required)

    def test__uncached_lookup_order_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertEqual(result, None)

    def test__uncached_lookup_extendors_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        subr = self._makeSubregistry()
        subr._adapters = [{}, {}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertEqual(result, None)

    def test__uncached_lookup_components_miss_wrong_iface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        IQux = InterfaceClass('IQux')
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        irrelevant = object()
        subr._adapters = [{}, {IFoo: {IQux: {'': irrelevant}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertEqual(result, None)

    def test__uncached_lookup_components_miss_wrong_name(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        wrongname = object()
        subr._adapters = [{}, {IFoo: {IBar: {'wrongname': wrongname}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertEqual(result, None)

    def test__uncached_lookup_simple_hit(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        subr._adapters = [{}, {IFoo: {IBar: {'': _expected}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookup((IFoo,), IBar)
        self.assertIs(result, _expected)

    def test__uncached_lookup_repeated_hit(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        subr._adapters = [{}, {IFoo: {IBar: {'': _expected}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookup((IFoo,), IBar)
        result2 = alb._uncached_lookup((IFoo,), IBar)
        self.assertIs(result, _expected)
        self.assertIs(result2, _expected)

    def test_queryMultiAdaptor_lookup_miss(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        registry = self._makeRegistry()
        subr = self._makeSubregistry()
        subr._adapters = [{}, {}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.lookup = alb._uncached_lookup
        subr._v_lookup = alb
        _default = object()
        result = alb.queryMultiAdapter((foo,), IBar, default=_default)
        self.assertIs(result, _default)

    def test_queryMultiAdapter_errors_on_attribute_access(self):
        from zope.interface.interface import InterfaceClass
        from zope.interface.tests import MissingSomeAttrs
        IFoo = InterfaceClass('IFoo')
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        alb.lookup = alb._uncached_lookup

        def test(ob):
            return alb.queryMultiAdapter((ob,), IFoo)
        MissingSomeAttrs.test_raises(self, test, expected_missing='__class__')

    def test_queryMultiAdaptor_factory_miss(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        _called_with = []

        def _factory(context):
            _called_with.append(context)
        subr._adapters = [{}, {IFoo: {IBar: {'': _factory}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.lookup = alb._uncached_lookup
        subr._v_lookup = alb
        _default = object()
        result = alb.queryMultiAdapter((foo,), IBar, default=_default)
        self.assertIs(result, _default)
        self.assertEqual(_called_with, [foo])

    def test_queryMultiAdaptor_factory_hit(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        _called_with = []

        def _factory(context):
            _called_with.append(context)
            return _expected
        subr._adapters = [{}, {IFoo: {IBar: {'': _factory}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.lookup = alb._uncached_lookup
        subr._v_lookup = alb
        _default = object()
        result = alb.queryMultiAdapter((foo,), IBar, default=_default)
        self.assertIs(result, _expected)
        self.assertEqual(_called_with, [foo])

    def test_queryMultiAdapter_super_unwraps(self):
        alb = self._makeOne(self._makeRegistry())

        def lookup(*args):
            return factory

        def factory(*args):
            return args
        alb.lookup = lookup
        objects = [super(), 42, 'abc', super()]
        result = alb.queryMultiAdapter(objects, None)
        self.assertEqual(result, (self, 42, 'abc', self))

    def test__uncached_lookupAll_empty_ro(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        result = alb._uncached_lookupAll((IFoo,), IBar)
        self.assertEqual(result, ())
        self.assertEqual(len(alb._required), 1)
        self.assertIn(IFoo.weakref(), alb._required)

    def test__uncached_lookupAll_order_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookupAll((IFoo,), IBar)
        self.assertEqual(result, ())

    def test__uncached_lookupAll_extendors_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        subr = self._makeSubregistry()
        subr._adapters = [{}, {}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookupAll((IFoo,), IBar)
        self.assertEqual(result, ())

    def test__uncached_lookupAll_components_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        IQux = InterfaceClass('IQux')
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        irrelevant = object()
        subr._adapters = [{}, {IFoo: {IQux: {'': irrelevant}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookupAll((IFoo,), IBar)
        self.assertEqual(result, ())

    def test__uncached_lookupAll_simple_hit(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        _named = object()
        subr._adapters = [{}, {IFoo: {IBar: {'': _expected, 'named': _named}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_lookupAll((IFoo,), IBar)
        self.assertEqual(sorted(result), [('', _expected), ('named', _named)])

    def test_names(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _expected = object()
        _named = object()
        subr._adapters = [{}, {IFoo: {IBar: {'': _expected, 'named': _named}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.lookupAll = alb._uncached_lookupAll
        subr._v_lookup = alb
        result = alb.names((IFoo,), IBar)
        self.assertEqual(sorted(result), ['', 'named'])

    def test__uncached_subscriptions_empty_ro(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        alb = self._makeOne(registry)
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(result, [])
        self.assertEqual(len(alb._required), 1)
        self.assertIn(IFoo.weakref(), alb._required)

    def test__uncached_subscriptions_order_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(result, [])

    def test__uncached_subscriptions_extendors_miss(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry()
        subr = self._makeSubregistry()
        subr._subscribers = [{}, {}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(result, [])

    def test__uncached_subscriptions_components_miss_wrong_iface(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        IQux = InterfaceClass('IQux')
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        irrelevant = object()
        subr._subscribers = [{}, {IFoo: {IQux: {'': irrelevant}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(result, [])

    def test__uncached_subscriptions_components_miss_wrong_name(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        wrongname = object()
        subr._subscribers = [{}, {IFoo: {IBar: {'wrongname': wrongname}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(result, [])

    def test__uncached_subscriptions_simple_hit(self):
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()

        class Foo:

            def __lt__(self, other):
                return True
        _exp1, _exp2 = (Foo(), Foo())
        subr._subscribers = [{}, {IFoo: {IBar: {'': (_exp1, _exp2)}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        subr._v_lookup = alb
        result = alb._uncached_subscriptions((IFoo,), IBar)
        self.assertEqual(sorted(result), sorted([_exp1, _exp2]))

    def test_subscribers_wo_provided(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        registry = self._makeRegistry(IFoo, IBar)
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _called = {}

        def _factory1(context):
            _called.setdefault('_factory1', []).append(context)

        def _factory2(context):
            _called.setdefault('_factory2', []).append(context)
        subr._subscribers = [{}, {IFoo: {None: {'': (_factory1, _factory2)}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.subscriptions = alb._uncached_subscriptions
        subr._v_lookup = alb
        result = alb.subscribers((foo,), None)
        self.assertEqual(result, ())
        self.assertEqual(_called, {'_factory1': [foo], '_factory2': [foo]})

    def test_subscribers_w_provided(self):
        from zope.interface.declarations import implementer
        from zope.interface.interface import InterfaceClass
        IFoo = InterfaceClass('IFoo')
        IBar = InterfaceClass('IBar', (IFoo,))

        @implementer(IFoo)
        class Foo:
            pass
        foo = Foo()
        registry = self._makeRegistry(IFoo, IBar)
        registry = self._makeRegistry(IFoo, IBar)
        subr = self._makeSubregistry()
        _called = {}
        _exp1, _exp2 = (object(), object())

        def _factory1(context):
            _called.setdefault('_factory1', []).append(context)
            return _exp1

        def _factory2(context):
            _called.setdefault('_factory2', []).append(context)
            return _exp2

        def _side_effect_only(context):
            _called.setdefault('_side_effect_only', []).append(context)
        subr._subscribers = [{}, {IFoo: {IBar: {'': (_factory1, _factory2, _side_effect_only)}}}]
        registry.ro.append(subr)
        alb = self._makeOne(registry)
        alb.subscriptions = alb._uncached_subscriptions
        subr._v_lookup = alb
        result = alb.subscribers((foo,), IBar)
        self.assertEqual(result, [_exp1, _exp2])
        self.assertEqual(_called, {'_factory1': [foo], '_factory2': [foo], '_side_effect_only': [foo]})