import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
class Test_taggedValue(unittest.TestCase):

    def test_w_single(self):
        from zope.interface.interface import TAGGED_DATA
        from zope.interface.interface import taggedValue

        class Foo:
            taggedValue('bar', ['baz'])
        self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'bar': ['baz']})

    def test_w_multiple(self):
        from zope.interface.interface import TAGGED_DATA
        from zope.interface.interface import taggedValue

        class Foo:
            taggedValue('bar', ['baz'])
            taggedValue('qux', 'spam')
        self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'bar': ['baz'], 'qux': 'spam'})

    def test_w_multiple_overwriting(self):
        from zope.interface.interface import TAGGED_DATA
        from zope.interface.interface import taggedValue

        class Foo:
            taggedValue('bar', ['baz'])
            taggedValue('qux', 'spam')
            taggedValue('bar', 'frob')
        self.assertEqual(getattr(Foo, TAGGED_DATA, None), {'bar': 'frob', 'qux': 'spam'})