import unittest
from traits.api import (
class TestReadOnlyTrait(unittest.TestCase):
    """ Test ReadOnly TraitType. """

    def test_set_readonly_trait_in_init(self):
        obj = ObjectWithReadOnlyText(text='ABC')
        self.assertEqual(obj.text, 'ABC')
        with self.assertRaises(TraitError):
            obj.text = 'XYZ'
        self.assertEqual(obj.text, 'ABC')