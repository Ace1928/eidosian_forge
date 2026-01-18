import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestAttrCompletion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.com = autocomplete.AttrCompletion()

    def test_att_matches_found_on_instance(self):
        self.assertSetEqual(self.com.matches(2, 'a.', locals_={'a': Foo()}), {'a.method', 'a.a', 'a.b'})

    def test_descriptor_attributes_not_run(self):
        com = autocomplete.AttrCompletion()
        self.assertSetEqual(com.matches(2, 'a.', locals_={'a': Properties()}), {'a.b', 'a.a', 'a.method', 'a.asserts_when_called'})

    def test_custom_get_attribute_not_invoked(self):
        com = autocomplete.AttrCompletion()
        self.assertSetEqual(com.matches(2, 'a.', locals_={'a': OverriddenGetattribute()}), {'a.b', 'a.a', 'a.method'})

    def test_slots_not_crash(self):
        com = autocomplete.AttrCompletion()
        self.assertSetEqual(com.matches(2, 'A.', locals_={'A': Slots}), {'A.b', 'A.a'})