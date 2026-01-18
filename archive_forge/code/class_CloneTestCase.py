import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
class CloneTestCase(unittest.TestCase):
    """ Test cases for traits clone """

    def test_any(self):
        b = ClassWithAny()
        f = Foo()
        f.s = 'the f'
        b.x = f
        bc = b.clone_traits(traits='all', copy='deep')
        self.assertNotEqual(id(bc.x), id(f), 'Foo x not cloned')

    def test_instance(self):
        b = ClassWithInstance()
        f = Foo()
        f.s = 'the f'
        b.x = f
        bc = b.clone_traits(traits='all', copy='deep')
        self.assertNotEqual(id(bc.x), id(f), 'Foo x not cloned')

    def test_class_attribute_missing(self):
        """ This test demonstrates a problem with Traits objects with class
        attributes.  A change to the value of a class attribute via one
        instance causes the attribute to be removed from other instances.

        AttributeError: 'ClassWithClassAttribute' object has no attribute
        'name'
        """
        s = 'class defined name'
        c = ClassWithClassAttribute()
        self.assertEqual(s, c.name)
        c2 = ClassWithClassAttribute()
        self.assertEqual(s, c.name)
        self.assertEqual(s, c2.name)
        s2 = 'name class attribute changed via clone'
        c2.name = s2
        self.assertEqual(s2, c2.name)
        self.assertEqual(s, c.name)

    def test_Any_circular_references(self):
        bar = BarAny()
        baz = BazAny()
        bar.other = baz
        baz.other = bar
        bar_copy = bar.clone_traits()
        self.assertIsNot(bar_copy, bar)
        self.assertIs(bar_copy.other, baz)
        self.assertIs(bar_copy.other.other, bar)

    def test_Any_circular_references_deep(self):
        bar = BarAny()
        baz = BazAny()
        bar.other = baz
        baz.other = bar
        bar_copy = bar.clone_traits(copy='deep')
        self.assertIsNot(bar_copy, bar)
        self.assertIsNot(bar_copy.other, baz)
        self.assertIsNot(bar_copy.other.other, bar)
        self.assertIs(bar_copy.other.other, bar_copy)

    def test_Instance_circular_references(self):
        ref = Foo(s='ref')
        bar_unique = Foo(s='bar.foo')
        shared = Foo(s='shared')
        baz_unique = Foo(s='baz.unique')
        baz = BazInstance()
        baz.unique = baz_unique
        baz.shared = shared
        baz.ref = ref
        bar = BarInstance()
        bar.unique = bar_unique
        bar.shared = shared
        bar.ref = ref
        bar.other = baz
        baz.other = bar
        baz_copy = baz.clone_traits()
        self.assertIsNot(baz_copy, baz)
        self.assertIsNot(baz_copy.other, bar)
        self.assertIsNot(baz_copy.unique, baz.unique)
        self.assertIsNot(baz_copy.shared, baz.shared)
        self.assertIs(baz_copy.ref, ref)
        bar_copy = baz_copy.other
        self.assertIsNot(bar_copy.unique, bar.unique)
        self.assertIs(bar_copy.ref, ref)
        self.assertIsNot(bar_copy.other, baz_copy)
        self.assertIs(bar_copy.other, baz)
        self.assertIsNot(bar_copy.shared, baz.shared)
        self.assertIs(bar_copy.shared, baz_copy.shared)

    def test_Instance_circular_references_deep(self):
        ref = Foo(s='ref')
        bar_unique = Foo(s='bar.foo')
        shared = Foo(s='shared')
        baz_unique = Foo(s='baz.unique')
        baz = BazInstance()
        baz.unique = baz_unique
        baz.shared = shared
        baz.ref = ref
        bar = BarInstance()
        bar.unique = bar_unique
        bar.shared = shared
        bar.ref = ref
        bar.other = baz
        baz.other = bar
        baz_copy = baz.clone_traits(copy='deep')
        self.assertIsNot(baz_copy, baz)
        self.assertIsNot(baz_copy.other, bar)
        self.assertIsNot(baz_copy.unique, baz.unique)
        self.assertIsNot(baz_copy.shared, baz.shared)
        bar_copy = baz_copy.other
        self.assertIsNot(bar_copy.unique, bar.unique)
        self.assertIs(baz_copy.ref, bar_copy.ref)
        self.assertIs(bar_copy.ref, ref)
        self.assertIsNot(bar_copy.other, baz_copy)
        self.assertIs(bar_copy.other, baz)
        self.assertIsNot(bar_copy.shared, baz.shared)
        self.assertIs(bar_copy.shared, baz_copy.shared)