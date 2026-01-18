import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
class TestTraitDict(unittest.TestCase):

    def setUp(self):
        self.added = None
        self.changed = None
        self.removed = None
        self.trait_dict = None

    def notification_handler(self, trait_dict, removed, added, changed):
        self.trait_list = trait_dict
        self.removed = removed
        self.added = added
        self.changed = changed

    def test_init(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator)
        self.assertEqual(td, {'a': 1, 'b': 2})
        self.assertEqual(td.notifiers, [])

    def test_init_iterable(self):
        td = TraitDict([('a', 1), ('b', 2)], key_validator=str_validator, value_validator=int_validator)
        self.assertEqual(td, {'a': 1, 'b': 2})
        self.assertEqual(td.notifiers, [])
        with self.assertRaises(ValueError):
            TraitDict(['a', 'b'], key_validator=str_validator, value_validator=int_validator)

    def test_notification(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td['c'] = 5
        self.assertEqual(self.added, {'c': 5})
        self.assertEqual(self.changed, {})
        self.assertEqual(self.removed, {})

    def test_deepcopy(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td_copy = copy.deepcopy(td)
        self.assertEqual(td, td_copy)
        self.assertEqual(td_copy.notifiers, [])
        self.assertEqual(td_copy.value_validator, td.value_validator)
        self.assertEqual(td_copy.key_validator, td.key_validator)

    def test_setitem(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td['a'] = 5
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {'a': 1})
        self.assertEqual(self.removed, {})
        with self.assertRaises(TraitError):
            td[5] = 'a'

    def test_delitem(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        del td['a']
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {})
        self.assertEqual(self.removed, {'a': 1})

    def test_delitem_not_found(self):
        python_dict = dict()
        with self.assertRaises(KeyError) as python_e:
            del python_dict['x']
        td = TraitDict()
        with self.assertRaises(KeyError) as trait_e:
            del td['x']
        self.assertEqual(str(trait_e.exception), str(python_e.exception))
    if sys.version_info >= (3, 9):

        def test_ior(self):
            td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
            td |= {'a': 3, 'd': 5}
            self.assertEqual(td, {'a': 3, 'b': 2, 'd': 5})
            self.assertEqual(self.added, {'d': 5})
            self.assertEqual(self.changed, {'a': 1})
            self.assertEqual(self.removed, {})

        def test_ior_is_quiet_if_no_change(self):
            td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
            td |= []
            self.assertEqual(td, {'a': 1, 'b': 2})
            self.assertIsNone(self.added)
            self.assertIsNone(self.removed)
            self.assertIsNone(self.changed)
    else:

        def test_ior(self):
            td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
            with self.assertRaises(TypeError):
                td |= {'a': 3, 'd': 5}

    def test_update(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td.update({'a': 2, 'b': 4, 'c': 5})
        self.assertEqual(self.added, {'c': 5})
        self.assertEqual(self.changed, {'a': 1, 'b': 2})
        self.assertEqual(self.removed, {})

    def test_update_iterable(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td.update([('a', 2), ('b', 4), ('c', 5)])
        self.assertEqual(self.added, {'c': 5})
        self.assertEqual(self.changed, {'a': 1, 'b': 2})
        self.assertEqual(self.removed, {})

    def test_update_with_transformation(self):
        td = TraitDict({'1': 1, '2': 2}, key_validator=str, notifiers=[self.notification_handler])
        td.update({1: 2})
        self.assertEqual(td, {'1': 2, '2': 2})
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {'1': 1})
        self.assertEqual(self.removed, {})

    def test_update_with_empty_argument(self):
        td = TraitDict({'1': 1, '2': 2}, key_validator=str, notifiers=[self.notification_handler])
        td.update([])
        td.update({})
        self.assertEqual(td, {'1': 1, '2': 2})
        self.assertIsNone(self.added)
        self.assertIsNone(self.changed)
        self.assertIsNone(self.removed)

    def test_update_notifies_with_nonempty_argument(self):
        td = TraitDict({'1': 1, '2': 2}, key_validator=str, notifiers=[self.notification_handler])
        td.update({'1': 1})
        self.assertEqual(td, {'1': 1, '2': 2})
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {'1': 1})
        self.assertEqual(self.removed, {})

    def test_clear(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td.clear()
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {})
        self.assertEqual(self.removed, {'a': 1, 'b': 2})

    def test_clear_empty_dictionary(self):
        td = TraitDict({}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td.clear()
        self.assertIsNone(self.added)
        self.assertIsNone(self.changed)
        self.assertIsNone(self.removed)

    def test_invalid_key(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        with self.assertRaises(TraitError):
            td[3] = '3'

    def test_invalid_value(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        with self.assertRaises(TraitError):
            td['3'] = True

    def test_setdefault(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        result = td.setdefault('c', 3)
        self.assertEqual(result, 3)
        self.assertEqual(td.setdefault('a', 5), 1)

    def test_setdefault_with_casting(self):
        notifier = mock.Mock()
        td = TraitDict(key_validator=str, value_validator=str, notifiers=[notifier, self.notification_handler])
        td.setdefault(1, 2)
        self.assertEqual(td, {'1': '2'})
        self.assertEqual(notifier.call_count, 1)
        self.assertEqual(self.removed, {})
        self.assertEqual(self.added, {'1': '2'})
        self.assertEqual(self.changed, {})
        notifier.reset_mock()
        td.setdefault(1, 4)
        self.assertEqual(td, {'1': '4'})
        self.assertEqual(notifier.call_count, 1)
        self.assertEqual(self.removed, {})
        self.assertEqual(self.added, {})
        self.assertEqual(self.changed, {'1': '2'})

    def test_pop(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        td.pop('b', 'X')
        self.assertEqual(self.removed, {'b': 2})
        self.removed = None
        res = td.pop('x', 'X')
        self.assertIsNone(self.removed)
        self.assertEqual(res, 'X')

    def test_pop_key_error(self):
        python_dict = {}
        with self.assertRaises(KeyError) as python_e:
            python_dict.pop('a')
        td = TraitDict()
        with self.assertRaises(KeyError) as trait_e:
            td.pop('a')
        self.assertEqual(str(trait_e.exception), str(python_e.exception))

    def test_popitem(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        items_cpy = td.copy().items()
        itm = td.popitem()
        self.assertIn(itm, items_cpy)
        self.assertNotIn(itm, td.items())
        td = TraitDict({}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        with self.assertRaises(KeyError):
            td.popitem()

    def test_pickle(self):
        td = TraitDict({'a': 1, 'b': 2}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            td_unpickled = pickle.loads(pickle.dumps(td, protocol=protocol))
            self.assertIs(td_unpickled.key_validator, str_validator)
            self.assertIs(td_unpickled.value_validator, int_validator)
            self.assertEqual(td_unpickled.notifiers, [])