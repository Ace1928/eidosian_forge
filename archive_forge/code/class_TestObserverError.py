import unittest
from traits.api import (
from traits.observation.api import (
class TestObserverError(unittest.TestCase):

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_trait_is_not_list(self):
        team = Team()
        team.observe(lambda e: None, trait('leader').list_items())
        person = Person()
        with self.assertRaises(ValueError) as exception_cm:
            team.leader = person
        self.assertIn('Expected a TraitList to be observed', str(exception_cm.exception))

    def test_items_on_a_list_not_observable_by_named_trait(self):
        team = Team()
        team.observe(lambda e: None, trait('member_names').list_items().trait('does_not_exist'))
        with self.assertRaises(ValueError) as exception_cm:
            team.member_names = ['Paul']
        self.assertEqual(str(exception_cm.exception), "Trait named 'does_not_exist' not found on 'Paul'.")

    def test_extended_trait_on_any_value(self):
        team = Team()
        team.any_value = 123
        with self.assertRaises(ValueError) as exception_cm:
            team.observe(lambda e: None, trait('any_value').trait('does_not_exist'))
        self.assertEqual(str(exception_cm.exception), "Trait named 'does_not_exist' not found on 123.")

    def test_no_new_trait_added(self):
        team = Team()
        team.observe(lambda e: None, trait('leader').trait('does_not_exist'))
        with self.assertRaises(ValueError):
            team.leader = Person()
        self.assertNotIn('does_not_exist', team.leader.trait_names())