import unittest
import warnings
from traits.api import (
class RichCompareTests:

    def bar_changed(self, object, trait, old, new):
        self.changed_object = object
        self.changed_trait = trait
        self.changed_old = old
        self.changed_new = new
        self.changed_count += 1

    def reset_change_tracker(self):
        self.changed_object = None
        self.changed_trait = None
        self.changed_old = None
        self.changed_new = None
        self.changed_count = 0

    def check_tracker(self, object, trait, old, new, count):
        self.assertEqual(count, self.changed_count)
        self.assertIs(object, self.changed_object)
        self.assertEqual(trait, self.changed_trait)
        self.assertIs(old, self.changed_old)
        self.assertIs(new, self.changed_new)

    def test_id_first_assignment(self):
        ic = IdentityCompare()
        ic.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker(ic, 'bar', default_value, self.a, 1)

    def test_rich_first_assignment(self):
        rich = RichCompare()
        rich.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)

    def test_id_same_object(self):
        ic = IdentityCompare()
        ic.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker(ic, 'bar', default_value, self.a, 1)
        ic.bar = self.a
        self.check_tracker(ic, 'bar', default_value, self.a, 1)

    def test_rich_same_object(self):
        rich = RichCompare()
        rich.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)
        rich.bar = self.a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)

    def test_id_different_object(self):
        ic = IdentityCompare()
        ic.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker(ic, 'bar', default_value, self.a, 1)
        ic.bar = self.different_from_a
        self.check_tracker(ic, 'bar', self.a, self.different_from_a, 2)

    def test_rich_different_object(self):
        rich = RichCompare()
        rich.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)
        rich.bar = self.different_from_a
        self.check_tracker(rich, 'bar', self.a, self.different_from_a, 2)

    def test_id_different_object_same_as(self):
        ic = IdentityCompare()
        ic.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = ic.bar
        ic.bar = self.a
        self.check_tracker(ic, 'bar', default_value, self.a, 1)
        ic.bar = self.same_as_a
        self.check_tracker(ic, 'bar', self.a, self.same_as_a, 2)

    def test_rich_different_object_same_as(self):
        rich = RichCompare()
        rich.on_trait_change(self.bar_changed, 'bar')
        self.reset_change_tracker()
        default_value = rich.bar
        rich.bar = self.a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)
        rich.bar = self.same_as_a
        self.check_tracker(rich, 'bar', default_value, self.a, 1)