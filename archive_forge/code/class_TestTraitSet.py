import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
class TestTraitSet(unittest.TestCase):

    def setUp(self):
        self.added = None
        self.removed = None
        self.validator_args = None
        self.trait_set = None

    def notification_handler(self, trait_set, removed, added):
        self.trait_set = trait_set
        self.removed = removed
        self.added = added

    def validator(self, added):
        self.validator_args = added
        return added

    def test_init(self):
        ts = TraitSet({1, 2, 3})
        self.assertEqual(ts, {1, 2, 3})
        self.assertIs(ts.item_validator, _validate_everything)
        self.assertEqual(ts.notifiers, [])

    def test_init_with_no_input(self):
        ts = TraitSet()
        self.assertEqual(ts, set())
        self.assertIs(ts.item_validator, _validate_everything)
        self.assertEqual(ts.notifiers, [])

    def test_validator(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator)
        self.assertEqual(ts, {1, 2, 3})
        self.assertEqual(ts.item_validator, int_validator)
        self.assertEqual(ts.notifiers, [])

    def test_notification(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        self.assertEqual(ts, {1, 2, 3})
        self.assertEqual(ts.item_validator, int_validator)
        self.assertEqual(ts.notifiers, [self.notification_handler])
        ts.add(5)
        self.assertEqual(ts, {1, 2, 3, 5})
        self.assertIs(self.trait_set, ts)
        self.assertEqual(self.removed, set())
        self.assertEqual(self.added, {5})

    def test_add(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts.add(5)
        self.assertEqual(ts, {1, 2, 3, 5})
        self.assertEqual(self.removed, set())
        self.assertEqual(self.added, {5})
        ts = TraitSet({'one', 'two', 'three'}, item_validator=string_validator, notifiers=[self.notification_handler])
        ts.add('four')
        self.assertEqual(ts, {'one', 'two', 'three', 'four'})
        self.assertEqual(self.removed, set())
        self.assertEqual(self.added, {'four'})

    def test_add_iterable(self):
        python_set = set()
        iterable = (i for i in range(4))
        python_set.add(iterable)
        ts = TraitSet()
        ts.add(iterable)
        next(iterable)
        self.assertEqual(ts, python_set)

    def test_add_unhashable(self):
        with self.assertRaises(TypeError) as python_e:
            set().add([])
        with self.assertRaises(TypeError) as trait_e:
            TraitSet().add([])
        self.assertEqual(str(trait_e.exception), str(python_e.exception))

    def test_add_no_notification_for_no_op(self):
        notifier = mock.Mock()
        ts = TraitSet({1, 2}, notifiers=[notifier])
        ts.add(1)
        notifier.assert_not_called()

    def test_remove(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts.remove(3)
        self.assertEqual(ts, {1, 2})
        self.assertEqual(self.removed, {3})
        self.assertEqual(self.added, set())
        with self.assertRaises(KeyError):
            ts.remove(3)

    def test_remove_iterable(self):
        iterable = (i for i in range(4))
        ts = TraitSet()
        ts.add(iterable)
        self.assertIn(iterable, ts)
        ts.remove(iterable)
        self.assertEqual(ts, set())

    def test_remove_does_not_call_validator(self):
        ts = TraitSet(item_validator=self.validator)
        ts.add('123')
        value, = ts
        self.validator_args = None
        ts.remove(value)
        self.assertIsNone(self.validator_args)

    def test_update_with_non_iterable(self):
        python_set = set()
        with self.assertRaises(TypeError) as python_exc:
            python_set.update(None)
        ts = TraitSet()
        with self.assertRaises(TypeError) as trait_exc:
            ts.update(None)
        self.assertEqual(str(trait_exc.exception), str(python_exc.exception))

    def test_update_varargs(self):
        ts = TraitSet(notifiers=[self.notification_handler])
        ts.update({1, 2}, {3, 4})
        self.assertEqual(self.added, {1, 2, 3, 4})
        self.assertEqual(self.removed, set())

    def test_update_with_nothing(self):
        notifier = mock.Mock()
        python_set = set()
        python_set.update()
        ts = TraitSet(notifiers=[notifier])
        ts.update()
        notifier.assert_not_called()
        self.assertEqual(ts, python_set)

    def test_discard(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts.discard(3)
        self.assertEqual(ts, {1, 2})
        self.assertEqual(self.removed, {3})
        self.assertEqual(self.added, set())
        ts.discard(3)

    def test_pop(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        val = ts.pop()
        self.assertIn(val, {1, 2, 3})
        self.assertEqual(self.removed, {val})
        self.assertEqual(self.added, set())

    def test_clear(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts.clear()
        self.assertEqual(self.removed, {1, 2, 3})
        self.assertEqual(self.added, set())

    def test_clear_no_notifications_if_already_empty(self):
        notifier = mock.Mock()
        ts = TraitSet(notifiers=[notifier])
        ts.clear()
        notifier.assert_not_called()

    def test_ior(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts |= {4, 5}
        self.assertEqual(self.removed, set())
        self.assertEqual(self.added, {4, 5})
        ts2 = TraitSet({6, 7}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts |= ts2
        self.assertEqual(self.removed, set())
        self.assertEqual(self.added, {6, 7})
        with self.assertRaises(TypeError):
            ts |= 8

    def test_iand(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts &= {1, 2, 3}
        self.assertIsNone(self.removed)
        self.assertIsNone(self.added)
        ts &= {1, 2}
        self.assertEqual(self.removed, {3})
        self.assertEqual(self.added, set())
        with self.assertRaises(TypeError):
            ts &= [3]

    def test_iand_does_not_call_validator(self):
        ts = TraitSet({1, 2, 3}, item_validator=self.validator)
        values = list(ts)
        python_set = set(ts)
        python_set &= set(values[:2])
        self.validator_args = None
        ts &= set(values[:2])
        self.assertEqual(ts, python_set)
        self.assertIsNone(self.validator_args)

    def test_intersection_update_with_no_arguments(self):
        python_set = set([1, 2, 3])
        python_set.intersection_update()
        notifier = mock.Mock()
        ts = TraitSet([1, 2, 3], notifiers=[notifier])
        ts.intersection_update()
        self.assertEqual(ts, python_set)
        notifier.assert_not_called

    def test_intersection_update_varargs(self):
        python_set = set([1, 2, 3])
        python_set.intersection_update([2], [3])
        ts = TraitSet([1, 2, 3])
        ts.intersection_update([2], [3])
        self.assertEqual(ts, python_set)

    def test_intersection_update_with_iterable(self):
        python_set = set([1, 2, 3])
        python_set.intersection_update((i for i in [1, 2]))
        ts = TraitSet([1, 2, 3])
        ts.intersection_update((i for i in [1, 2]))
        self.assertEqual(ts, python_set)

    def test_ixor(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts ^= {1, 2, 3, 5}
        self.assertEqual(self.removed, {1, 2, 3})
        self.assertEqual(self.added, {5})
        self.assertEqual(ts, {5})
        with self.assertRaises(TypeError):
            ts ^= [5]

    def test_ixor_no_nofications_for_no_change(self):
        notifier = mock.Mock()
        ts_1 = TraitSet([1, 2], notifiers=[notifier])
        ts_1 ^= set()
        notifier.assert_not_called()

    def test_ixor_with_iterable_items(self):
        iterable = range(2)
        python_set = set([iterable])
        python_set ^= set([iterable])
        self.assertEqual(python_set, set())
        ts = TraitSet([iterable], item_validator=self.validator)
        self.validator_args = None
        ts ^= {iterable}
        self.assertEqual(ts, set())
        self.assertIsNone(self.validator_args)

    def test_ixor_validator_args_with_added(self):
        validator = mock.Mock(wraps=str)
        ts = TraitSet([1, 2, 3], item_validator=validator, notifiers=[self.notification_handler])
        self.assertEqual(ts, set(['1', '2', '3']))
        validator.reset_mock()
        ts ^= set(['2', 3, 4])
        validator_inputs = set((value for (value,), _ in validator.call_args_list))
        self.assertEqual(validator_inputs, set([3, 4]))
        self.assertEqual(ts, set(['1', '3', '4']))
        self.assertEqual(self.added, set(['4']))
        self.assertEqual(self.removed, set(['2']))

    def test_isub(self):
        ts = TraitSet({1, 2, 3}, item_validator=int_validator, notifiers=[self.notification_handler])
        ts -= {2, 3, 5}
        self.assertEqual(self.removed, {2, 3})
        self.assertEqual(self.added, set())
        self.assertEqual(ts, {1})
        with self.assertRaises(TypeError):
            ts -= [4, 5]

    def test_isub_validator_not_called(self):
        ts = TraitSet({1, 2, 3}, item_validator=self.validator)
        values = set(ts)
        self.validator_args = None
        ts -= values
        self.assertIsNone(self.validator_args)

    def test_isub_with_no_intersection(self):
        python_set = set([3, 4, 5])
        python_set -= set((i for i in range(2)))
        notifier = mock.Mock()
        ts = TraitSet((3, 4, 5), notifiers=[notifier])
        ts -= set((i for i in range(2)))
        self.assertEqual(ts, python_set)
        notifier.assert_not_called()

    def test_difference_update_with_no_arguments(self):
        python_set = set([1, 2, 3])
        python_set.difference_update()
        ts = TraitSet([1, 2, 3])
        ts.difference_update()
        self.assertEqual(ts, python_set)

    def test_difference_update_varargs(self):
        ts = TraitSet([1, 2, 3], notifiers=[self.notification_handler])
        ts.difference_update([2], [3])
        self.assertEqual(self.removed, {2, 3})

    def test_get_state(self):
        ts = TraitSet(notifiers=[self.notification_handler])
        states = ts.__getstate__()
        self.assertNotIn('notifiers', states)

    def test_set_state_exclude_notifiers(self):
        ts = TraitSet(notifiers=[])
        ts.__setstate__({'notifiers': [self.notification_handler]})
        self.assertEqual(ts.notifiers, [])