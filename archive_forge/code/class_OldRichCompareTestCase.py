import unittest
import warnings
from traits.api import (
class OldRichCompareTestCase(unittest.TestCase):

    def test_rich_compare_false(self):
        with warnings.catch_warnings(record=True) as warn_msgs:
            warnings.simplefilter('always', DeprecationWarning)

            class OldRichCompare(HasTraits):
                bar = Any(rich_compare=False)
        self.assertEqual(len(warn_msgs), 1)
        warn_msg = warn_msgs[0]
        self.assertIs(warn_msg.category, DeprecationWarning)
        self.assertIn("'rich_compare' metadata has been deprecated", str(warn_msg.message))
        _, _, this_module = __name__.rpartition('.')
        self.assertIn(this_module, warn_msg.filename)
        old_compare = OldRichCompare()
        events = []
        old_compare.on_trait_change(lambda: events.append(None), 'bar')
        some_list = [1, 2, 3]
        self.assertEqual(len(events), 0)
        old_compare.bar = some_list
        self.assertEqual(len(events), 1)
        old_compare.bar = some_list
        self.assertEqual(len(events), 1)
        old_compare.bar = [1, 2, 3]
        self.assertEqual(len(events), 2)
        old_compare.bar = [4, 5, 6]
        self.assertEqual(len(events), 3)

    def test_rich_compare_true(self):
        with warnings.catch_warnings(record=True) as warn_msgs:
            warnings.simplefilter('always', DeprecationWarning)

            class OldRichCompare(HasTraits):
                bar = Any(rich_compare=True)
        self.assertEqual(len(warn_msgs), 1)
        warn_msg = warn_msgs[0]
        self.assertIs(warn_msg.category, DeprecationWarning)
        self.assertIn("'rich_compare' metadata has been deprecated", str(warn_msg.message))
        _, _, this_module = __name__.rpartition('.')
        self.assertIn(this_module, warn_msg.filename)
        old_compare = OldRichCompare()
        events = []
        old_compare.on_trait_change(lambda: events.append(None), 'bar')
        some_list = [1, 2, 3]
        self.assertEqual(len(events), 0)
        old_compare.bar = some_list
        self.assertEqual(len(events), 1)
        old_compare.bar = some_list
        self.assertEqual(len(events), 1)
        old_compare.bar = [1, 2, 3]
        self.assertEqual(len(events), 1)
        old_compare.bar = [4, 5, 6]
        self.assertEqual(len(events), 2)

    def test_rich_compare_with_cached_property(self):

        class Model(HasTraits):
            value = Property(depends_on='name')
            name = Str(comparison_mode=ComparisonMode.none)

            @cached_property
            def _get_value(self):
                return self.trait_names
        instance = Model()
        events = []
        instance.on_trait_change(lambda: events.append(None), 'value')
        instance.name = 'A'
        events.clear()
        instance.name = 'A'
        self.assertEqual(len(events), 1)