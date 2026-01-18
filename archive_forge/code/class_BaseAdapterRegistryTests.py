import unittest
from zope.interface.tests import OptimizationTestMixin
class BaseAdapterRegistryTests(unittest.TestCase):
    maxDiff = None

    def _getBaseAdapterRegistry(self):
        from zope.interface.adapter import BaseAdapterRegistry
        return BaseAdapterRegistry

    def _getTargetClass(self):
        BaseAdapterRegistry = self._getBaseAdapterRegistry()

        class _CUT(BaseAdapterRegistry):

            class LookupClass:
                _changed = _extendors = ()

                def __init__(self, reg):
                    pass

                def changed(self, orig):
                    self._changed += (orig,)

                def add_extendor(self, provided):
                    self._extendors += (provided,)

                def remove_extendor(self, provided):
                    self._extendors = tuple([x for x in self._extendors if x != provided])
        for name in BaseAdapterRegistry._delegated:
            setattr(_CUT.LookupClass, name, object())
        return _CUT

    def _makeOne(self):
        return self._getTargetClass()()

    def _getMappingType(self):
        return dict

    def _getProvidedType(self):
        return dict

    def _getMutableListType(self):
        return list

    def _getLeafSequenceType(self):
        return tuple

    def test_lookup_delegation(self):
        CUT = self._getTargetClass()
        registry = CUT()
        for name in CUT._delegated:
            self.assertIs(getattr(registry, name), getattr(registry._v_lookup, name))

    def test__generation_on_first_creation(self):
        registry = self._makeOne()
        self.assertEqual(registry._generation, 1)

    def test__generation_after_calling_changed(self):
        registry = self._makeOne()
        orig = object()
        registry.changed(orig)
        self.assertEqual(registry._generation, 2)
        self.assertEqual(registry._v_lookup._changed, (registry, orig))

    def test__generation_after_changing___bases__(self):

        class _Base:
            pass
        registry = self._makeOne()
        registry.__bases__ = (_Base,)
        self.assertEqual(registry._generation, 2)

    def _check_basic_types_of_adapters(self, registry, expected_order=2):
        self.assertEqual(len(registry._adapters), expected_order)
        self.assertIsInstance(registry._adapters, self._getMutableListType())
        MT = self._getMappingType()
        for mapping in registry._adapters:
            self.assertIsInstance(mapping, MT)
        self.assertEqual(registry._adapters[0], MT())
        self.assertIsInstance(registry._adapters[1], MT)
        self.assertEqual(len(registry._adapters[expected_order - 1]), 1)

    def _check_basic_types_of_subscribers(self, registry, expected_order=2):
        self.assertEqual(len(registry._subscribers), expected_order)
        self.assertIsInstance(registry._subscribers, self._getMutableListType())
        MT = self._getMappingType()
        for mapping in registry._subscribers:
            self.assertIsInstance(mapping, MT)
        if expected_order:
            self.assertEqual(registry._subscribers[0], MT())
            self.assertIsInstance(registry._subscribers[1], MT)
            self.assertEqual(len(registry._subscribers[expected_order - 1]), 1)

    def test_register(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([IB0], IR0, '', 'A1')
        self.assertEqual(registry.registered([IB0], IR0, ''), 'A1')
        self.assertEqual(registry._generation, 2)
        self._check_basic_types_of_adapters(registry)
        MT = self._getMappingType()
        self.assertEqual(registry._adapters[1], MT({IB0: MT({IR0: MT({'': 'A1'})})}))
        PT = self._getProvidedType()
        self.assertEqual(registry._provided, PT({IR0: 1}))
        registered = list(registry.allRegistrations())
        self.assertEqual(registered, [((IB0,), IR0, '', 'A1')])

    def test_register_multiple_allRegistrations(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([], IR0, '', 'A1')
        registry.register([], IR0, 'name1', 'A2')
        registry.register([IB0], IR0, '', 'A1')
        registry.register([IB0], IR0, 'name2', 'A2')
        registry.register([IB0], IR1, '', 'A3')
        registry.register([IB0], IR1, 'name3', 'A4')
        registry.register([IB0, IB1], IR0, '', 'A1')
        registry.register([IB0, IB2], IR0, 'name2', 'A2')
        registry.register([IB0, IB2], IR1, 'name4', 'A4')
        registry.register([IB0, IB3], IR1, '', 'A3')

        def build_adapters(L, MT):
            return L([MT({IR0: MT({'': 'A1', 'name1': 'A2'})}), MT({IB0: MT({IR0: MT({'': 'A1', 'name2': 'A2'}), IR1: MT({'': 'A3', 'name3': 'A4'})})}), MT({IB0: MT({IB1: MT({IR0: MT({'': 'A1'})}), IB2: MT({IR0: MT({'name2': 'A2'}), IR1: MT({'name4': 'A4'})}), IB3: MT({IR1: MT({'': 'A3'})})})})])
        self.assertEqual(registry._adapters, build_adapters(L=self._getMutableListType(), MT=self._getMappingType()))
        registered = sorted(registry.allRegistrations())
        self.assertEqual(registered, [((), IR0, '', 'A1'), ((), IR0, 'name1', 'A2'), ((IB0,), IR0, '', 'A1'), ((IB0,), IR0, 'name2', 'A2'), ((IB0,), IR1, '', 'A3'), ((IB0,), IR1, 'name3', 'A4'), ((IB0, IB1), IR0, '', 'A1'), ((IB0, IB2), IR0, 'name2', 'A2'), ((IB0, IB2), IR1, 'name4', 'A4'), ((IB0, IB3), IR1, '', 'A3')])
        registry2 = self._makeOne()
        for args in registered:
            registry2.register(*args)
        self.assertEqual(registry2._adapters, registry._adapters)
        self.assertEqual(registry2._provided, registry._provided)
        registry._mappingType = CustomMapping
        registry._leafSequenceType = CustomLeafSequence
        registry._sequenceType = CustomSequence
        registry._providedType = CustomProvided

        def addValue(existing, new):
            existing = existing if existing is not None else CustomLeafSequence()
            existing.append(new)
            return existing
        registry._addValueToLeaf = addValue
        registry.rebuild()
        self.assertEqual(registry._adapters, build_adapters(L=CustomSequence, MT=CustomMapping))

    def test_register_with_invalid_name(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        with self.assertRaises(ValueError):
            registry.register([IB0], IR0, object(), 'A1')

    def test_register_with_value_None_unregisters(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([None], IR0, '', 'A1')
        registry.register([None], IR0, '', None)
        self.assertEqual(len(registry._adapters), 0)
        self.assertIsInstance(registry._adapters, self._getMutableListType())
        registered = list(registry.allRegistrations())
        self.assertEqual(registered, [])

    def test_register_with_same_value(self):
        from zope.interface import Interface
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        _value = object()
        registry.register([None], IR0, '', _value)
        _before = registry._generation
        registry.register([None], IR0, '', _value)
        self.assertEqual(registry._generation, _before)
        self._check_basic_types_of_adapters(registry)
        MT = self._getMappingType()
        self.assertEqual(registry._adapters[1], MT({Interface: MT({IR0: MT({'': _value})})}))
        registered = list(registry.allRegistrations())
        self.assertEqual(registered, [((Interface,), IR0, '', _value)])

    def test_registered_empty(self):
        registry = self._makeOne()
        self.assertEqual(registry.registered([None], None, ''), None)
        registered = list(registry.allRegistrations())
        self.assertEqual(registered, [])

    def test_registered_non_empty_miss(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([IB1], None, '', 'A1')
        self.assertEqual(registry.registered([IB2], None, ''), None)

    def test_registered_non_empty_hit(self):
        registry = self._makeOne()
        registry.register([None], None, '', 'A1')
        self.assertEqual(registry.registered([None], None, ''), 'A1')

    def test_unregister_empty(self):
        registry = self._makeOne()
        registry.unregister([None], None, '')
        self.assertEqual(registry.registered([None], None, ''), None)
        self.assertEqual(len(registry._provided), 0)

    def test_unregister_non_empty_miss_on_required(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([IB1], None, '', 'A1')
        registry.unregister([IB2], None, '')
        self.assertEqual(registry.registered([IB1], None, ''), 'A1')
        self._check_basic_types_of_adapters(registry)
        MT = self._getMappingType()
        self.assertEqual(registry._adapters[1], MT({IB1: MT({None: MT({'': 'A1'})})}))
        PT = self._getProvidedType()
        self.assertEqual(registry._provided, PT({None: 1}))

    def test_unregister_non_empty_miss_on_name(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.register([IB1], None, '', 'A1')
        registry.unregister([IB1], None, 'nonesuch')
        self.assertEqual(registry.registered([IB1], None, ''), 'A1')
        self._check_basic_types_of_adapters(registry)
        MT = self._getMappingType()
        self.assertEqual(registry._adapters[1], MT({IB1: MT({None: MT({'': 'A1'})})}))
        PT = self._getProvidedType()
        self.assertEqual(registry._provided, PT({None: 1}))

    def test_unregister_with_value_not_None_miss(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        orig = object()
        nomatch = object()
        registry.register([IB1], None, '', orig)
        registry.unregister([IB1], None, '', nomatch)
        self.assertIs(registry.registered([IB1], None, ''), orig)

    def test_unregister_hit_clears_empty_subcomponents(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        one = object()
        another = object()
        registry.register([IB1, IB2], None, '', one)
        registry.register([IB1, IB3], None, '', another)
        self._check_basic_types_of_adapters(registry, expected_order=3)
        self.assertIn(IB2, registry._adapters[2][IB1])
        self.assertIn(IB3, registry._adapters[2][IB1])
        MT = self._getMappingType()
        self.assertEqual(registry._adapters[2], MT({IB1: MT({IB2: MT({None: MT({'': one})}), IB3: MT({None: MT({'': another})})})}))
        PT = self._getProvidedType()
        self.assertEqual(registry._provided, PT({None: 2}))
        registry.unregister([IB1, IB3], None, '', another)
        self.assertIn(IB2, registry._adapters[2][IB1])
        self.assertNotIn(IB3, registry._adapters[2][IB1])
        self.assertEqual(registry._adapters[2], MT({IB1: MT({IB2: MT({None: MT({'': one})})})}))
        self.assertEqual(registry._provided, PT({None: 1}))

    def test_unsubscribe_empty(self):
        registry = self._makeOne()
        registry.unsubscribe([None], None, '')
        self.assertEqual(registry.registered([None], None, ''), None)
        self._check_basic_types_of_subscribers(registry, expected_order=0)

    def test_unsubscribe_hit(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        orig = object()
        registry.subscribe([IB1], None, orig)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        PT = self._getProvidedType()
        self._check_basic_types_of_subscribers(registry)
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((orig,))})})}))
        self.assertEqual(registry._provided, PT({}))
        registry.unsubscribe([IB1], None, orig)
        self.assertEqual(len(registry._subscribers), 0)
        self.assertEqual(registry._provided, PT({}))

    def assertLeafIdentity(self, leaf1, leaf2):
        """
        Implementations may choose to use new, immutable objects
        instead of mutating existing subscriber leaf objects, or vice versa.

        The default implementation uses immutable tuples, so they are never
        the same. Other implementations may use persistent lists so they should be
        the same and mutated in place. Subclasses testing this behaviour need to
        override this method.
        """
        self.assertIsNot(leaf1, leaf2)

    def test_unsubscribe_after_multiple(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        first = object()
        second = object()
        third = object()
        fourth = object()
        registry.subscribe([IB1], None, first)
        registry.subscribe([IB1], None, second)
        registry.subscribe([IB1], IR0, third)
        registry.subscribe([IB1], IR0, fourth)
        self._check_basic_types_of_subscribers(registry, expected_order=2)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        PT = self._getProvidedType()
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((first, second))}), IR0: MT({'': L((third, fourth))})})}))
        self.assertEqual(registry._provided, PT({IR0: 2}))
        IR0_leaf_orig = registry._subscribers[1][IB1][IR0]['']
        Non_leaf_orig = registry._subscribers[1][IB1][None]['']
        registry.unsubscribe([IB1], None, first)
        registry.unsubscribe([IB1], IR0, third)
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((second,))}), IR0: MT({'': L((fourth,))})})}))
        self.assertEqual(registry._provided, PT({IR0: 1}))
        IR0_leaf_new = registry._subscribers[1][IB1][IR0]['']
        Non_leaf_new = registry._subscribers[1][IB1][None]['']
        self.assertLeafIdentity(IR0_leaf_orig, IR0_leaf_new)
        self.assertLeafIdentity(Non_leaf_orig, Non_leaf_new)
        registry.unsubscribe([IB1], None, second)
        registry.unsubscribe([IB1], IR0, fourth)
        self.assertEqual(len(registry._subscribers), 0)
        self.assertEqual(len(registry._provided), 0)

    def test_subscribe_unsubscribe_identical_objects_provided(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        first = object()
        registry.subscribe([IB1], IR0, first)
        registry.subscribe([IB1], IR0, first)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        PT = self._getProvidedType()
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({IR0: MT({'': L((first, first))})})}))
        self.assertEqual(registry._provided, PT({IR0: 2}))
        registry.unsubscribe([IB1], IR0, first)
        registry.unsubscribe([IB1], IR0, first)
        self.assertEqual(len(registry._subscribers), 0)
        self.assertEqual(registry._provided, PT())

    def test_subscribe_unsubscribe_nonequal_objects_provided(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        first = object()
        second = object()
        registry.subscribe([IB1], IR0, first)
        registry.subscribe([IB1], IR0, second)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        PT = self._getProvidedType()
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({IR0: MT({'': L((first, second))})})}))
        self.assertEqual(registry._provided, PT({IR0: 2}))
        registry.unsubscribe([IB1], IR0, first)
        registry.unsubscribe([IB1], IR0, second)
        self.assertEqual(len(registry._subscribers), 0)
        self.assertEqual(registry._provided, PT())

    def test_subscribed_empty(self):
        registry = self._makeOne()
        self.assertIsNone(registry.subscribed([None], None, ''))
        subscribed = list(registry.allSubscriptions())
        self.assertEqual(subscribed, [])

    def test_subscribed_non_empty_miss(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.subscribe([IB1], IF0, 'A1')
        self.assertIsNone(registry.subscribed([IB2], IF0, ''))
        self.assertIsNone(registry.subscribed([IB1], IF1, ''))
        self.assertIsNone(registry.subscribed([IB1], IF0, ''))

    def test_subscribed_non_empty_hit(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.subscribe([IB0], IF0, 'A1')
        self.assertEqual(registry.subscribed([IB0], IF0, 'A1'), 'A1')

    def test_unsubscribe_w_None_after_multiple(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        first = object()
        second = object()
        registry.subscribe([IB1], None, first)
        registry.subscribe([IB1], None, second)
        self._check_basic_types_of_subscribers(registry, expected_order=2)
        registry.unsubscribe([IB1], None)
        self.assertEqual(len(registry._subscribers), 0)

    def test_unsubscribe_non_empty_miss_on_required(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.subscribe([IB1], None, 'A1')
        self._check_basic_types_of_subscribers(registry, expected_order=2)
        registry.unsubscribe([IB2], None, '')
        self.assertEqual(len(registry._subscribers), 2)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L(('A1',))})})}))

    def test_unsubscribe_non_empty_miss_on_value(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.subscribe([IB1], None, 'A1')
        self._check_basic_types_of_subscribers(registry, expected_order=2)
        registry.unsubscribe([IB1], None, 'A2')
        self.assertEqual(len(registry._subscribers), 2)
        MT = self._getMappingType()
        L = self._getLeafSequenceType()
        self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L(('A1',))})})}))

    def test_unsubscribe_with_value_not_None_miss(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        orig = object()
        nomatch = object()
        registry.subscribe([IB1], None, orig)
        registry.unsubscribe([IB1], None, nomatch)
        self.assertEqual(len(registry._subscribers), 2)

    def _instance_method_notify_target(self):
        self.fail('Example method, not intended to be called.')

    def test_unsubscribe_instance_method(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        self.assertEqual(len(registry._subscribers), 0)
        registry.subscribe([IB1], None, self._instance_method_notify_target)
        registry.unsubscribe([IB1], None, self._instance_method_notify_target)
        self.assertEqual(len(registry._subscribers), 0)

    def test_subscribe_multiple_allRegistrations(self):
        IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
        registry = self._makeOne()
        registry.subscribe([], IR0, 'A1')
        registry.subscribe([], IR0, 'A2')
        registry.subscribe([IB0], IR0, 'A1')
        registry.subscribe([IB0], IR0, 'A2')
        registry.subscribe([IB0], IR1, 'A3')
        registry.subscribe([IB0], IR1, 'A4')
        registry.subscribe([IB0, IB1], IR0, 'A1')
        registry.subscribe([IB0, IB2], IR0, 'A2')
        registry.subscribe([IB0, IB2], IR1, 'A4')
        registry.subscribe([IB0, IB3], IR1, 'A3')

        def build_subscribers(L, F, MT):
            return L([MT({IR0: MT({'': F(['A1', 'A2'])})}), MT({IB0: MT({IR0: MT({'': F(['A1', 'A2'])}), IR1: MT({'': F(['A3', 'A4'])})})}), MT({IB0: MT({IB1: MT({IR0: MT({'': F(['A1'])})}), IB2: MT({IR0: MT({'': F(['A2'])}), IR1: MT({'': F(['A4'])})}), IB3: MT({IR1: MT({'': F(['A3'])})})})})])
        self.assertEqual(registry._subscribers, build_subscribers(L=self._getMutableListType(), F=self._getLeafSequenceType(), MT=self._getMappingType()))

        def build_provided(P):
            return P({IR0: 6, IR1: 4})
        self.assertEqual(registry._provided, build_provided(P=self._getProvidedType()))
        registered = sorted(registry.allSubscriptions())
        self.assertEqual(registered, [((), IR0, 'A1'), ((), IR0, 'A2'), ((IB0,), IR0, 'A1'), ((IB0,), IR0, 'A2'), ((IB0,), IR1, 'A3'), ((IB0,), IR1, 'A4'), ((IB0, IB1), IR0, 'A1'), ((IB0, IB2), IR0, 'A2'), ((IB0, IB2), IR1, 'A4'), ((IB0, IB3), IR1, 'A3')])
        registry2 = self._makeOne()
        for args in registered:
            registry2.subscribe(*args)
        self.assertEqual(registry2._subscribers, registry._subscribers)
        self.assertEqual(registry2._provided, registry._provided)
        registry._mappingType = CustomMapping
        registry._leafSequenceType = CustomLeafSequence
        registry._sequenceType = CustomSequence
        registry._providedType = CustomProvided

        def addValue(existing, new):
            existing = existing if existing is not None else CustomLeafSequence()
            existing.append(new)
            return existing
        registry._addValueToLeaf = addValue
        registry.rebuild()
        self.assertEqual(registry._subscribers, build_subscribers(L=CustomSequence, F=CustomLeafSequence, MT=CustomMapping))