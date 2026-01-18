import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestExpressionAttributeCompletion(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.com = autocomplete.ExpressionAttributeCompletion()

    def test_att_matches_found_on_instance(self):
        self.assertSetEqual(self.com.matches(5, 'a[0].', locals_={'a': [Foo()]}), {'method', 'a', 'b'})

    def test_other_getitem_methods_not_called(self):

        class FakeList:

            def __getitem__(inner_self, i):
                self.fail('possibly side-effecting __getitem_ method called')
        self.com.matches(5, 'a[0].', locals_={'a': FakeList()})

    def test_tuples_complete(self):
        self.assertSetEqual(self.com.matches(5, 'a[0].', locals_={'a': (Foo(),)}), {'method', 'a', 'b'})

    @unittest.skip('TODO, subclasses do not complete yet')
    def test_list_subclasses_complete(self):

        class ListSubclass(list):
            pass
        self.assertSetEqual(self.com.matches(5, 'a[0].', locals_={'a': ListSubclass([Foo()])}), {'method', 'a', 'b'})

    def test_getitem_not_called_in_list_subclasses_overriding_getitem(self):

        class FakeList(list):

            def __getitem__(inner_self, i):
                self.fail('possibly side-effecting __getitem_ method called')
        self.com.matches(5, 'a[0].', locals_={'a': FakeList()})

    def test_literals_complete(self):
        self.assertSetEqual(self.com.matches(10, '[a][0][0].', locals_={'a': (Foo(),)}), {'method', 'a', 'b'})

    def test_dictionaries_complete(self):
        self.assertSetEqual(self.com.matches(7, 'a["b"].', locals_={'a': {'b': Foo()}}), {'method', 'a', 'b'})