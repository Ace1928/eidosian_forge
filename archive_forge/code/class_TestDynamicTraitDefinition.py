import unittest
from traits.api import Float, HasTraits, Int, List
class TestDynamicTraitDefinition(unittest.TestCase):
    """ Test demonstrating special change events using the 'event' metadata.
    """

    def test_add_trait(self):
        foo = Foo(x=3)
        foo.add_trait('y', Int)
        self.assertTrue(hasattr(foo, 'y'))
        self.assertEqual(type(foo.y), int)
        foo.y = 4
        self.assertEqual(foo.y_changes, [4])

    def test_remove_trait(self):
        foo = Foo(x=3)
        result = foo.remove_trait('x')
        self.assertFalse(result)
        foo.add_trait('y', Int)
        foo.y = 70
        result = foo.remove_trait('y')
        self.assertTrue(result)
        self.assertFalse(hasattr(foo, 'y'))
        foo.y = 10
        self.assertEqual(foo.y_changes, [70])