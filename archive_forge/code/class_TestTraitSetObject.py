import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
class TestTraitSetObject(unittest.TestCase):

    def test_get_state(self):
        foo = Foo(values={1, 2, 3})
        self.assertEqual(foo.values.notifiers, [foo.values.notifier])
        states = foo.values.__getstate__()
        self.assertNotIn('notifiers', states)

    def test_pickle_with_notifier(self):
        foo = Foo(values={1, 2, 3})
        foo.values.notifiers.append(notifier)
        protocols = range(pickle.HIGHEST_PROTOCOL + 1)
        for protocol in protocols:
            with self.subTest(protocol=protocol):
                serialized = pickle.dumps(foo.values, protocol=protocol)
                deserialized = pickle.loads(serialized)
                self.assertEqual(deserialized.notifiers, [deserialized.notifier])

    def test_validation(self):

        class TestSet(HasTraits):
            letters = Set(Str())
        TestSet(letters={'4'})
        with self.assertRaises(TraitError):
            TestSet(letters={4})

    def test_notification_silenced_if_has_items_if_false(self):

        class Foo(HasTraits):
            values = Set(items=False)
        foo = Foo(values=set())
        notifier = mock.Mock()
        foo.on_trait_change(lambda: notifier(), 'values_items')
        foo.values.add(1)
        notifier.assert_not_called()