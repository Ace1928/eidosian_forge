import unittest
from traits.api import (
from traits.observation.api import (
class TestObserveAnytrait(unittest.TestCase):

    def test_observe_method_anytrait(self):
        obj = HasVariousTraits()
        events = []
        obj.observe(events.append, '*')
        obj.foo = 23
        obj.bar = 'on'
        self.assertEqual(len(events), 2)
        foo_event, bar_event = events
        self.assertEqual(foo_event.object, obj)
        self.assertEqual(foo_event.name, 'foo')
        self.assertEqual(foo_event.old, 16)
        self.assertEqual(foo_event.new, 23)
        self.assertEqual(bar_event.object, obj)
        self.assertEqual(bar_event.name, 'bar')
        self.assertEqual(bar_event.old, 'off')
        self.assertEqual(bar_event.new, 'on')

    def test_observe_decorator_anytrait(self):
        events = []
        obj = HasVariousTraits(trait_change_callback=events.append)
        obj.foo = 23
        obj.bar = 'on'
        self.assertEqual(len(events), 3)
        callback_event, foo_event, bar_event = events
        self.assertEqual(callback_event.object, obj)
        self.assertEqual(callback_event.name, 'trait_change_callback')
        self.assertIs(callback_event.old, None)
        self.assertEqual(callback_event.new, events.append)
        self.assertEqual(foo_event.object, obj)
        self.assertEqual(foo_event.name, 'foo')
        self.assertEqual(foo_event.old, 16)
        self.assertEqual(foo_event.new, 23)
        self.assertEqual(bar_event.object, obj)
        self.assertEqual(bar_event.name, 'bar')
        self.assertEqual(bar_event.old, 'off')
        self.assertEqual(bar_event.new, 'on')

    def test_anytrait_expression(self):
        obj = HasVariousTraits()
        events = []
        obj.observe(events.append, anytrait())
        obj.foo = 23
        obj.bar = 'on'
        self.assertEqual(len(events), 2)
        foo_event, bar_event = events
        self.assertEqual(foo_event.object, obj)
        self.assertEqual(foo_event.name, 'foo')
        self.assertEqual(foo_event.old, 16)
        self.assertEqual(foo_event.new, 23)
        self.assertEqual(bar_event.object, obj)
        self.assertEqual(bar_event.name, 'bar')
        self.assertEqual(bar_event.old, 'off')
        self.assertEqual(bar_event.new, 'on')

    def test_anytrait_method(self):
        foo = HasVariousTraits()
        bar = HasVariousTraits()
        obj = UpdateListener(foo=foo, bar=bar)
        events = []
        obj.observe(events.append, trait('foo', notify=False).anytrait())
        foo.updated = True
        bar.updated = True
        self.assertEqual(len(events), 1)
        foo_event, = events
        self.assertEqual(foo_event.object, foo)
        self.assertEqual(foo_event.name, 'updated')
        self.assertEqual(foo_event.old, Undefined)
        self.assertEqual(foo_event.new, True)

    def test_anytrait_with_children(self):
        foo = HasVariousTraits()
        bar = HasVariousTraits()
        obj = UpdateListener(foo=foo, bar=bar)
        events = []
        with self.assertRaises(ValueError):
            obj.observe(events.append, '*:updated')

    def test_anytrait_of_anytrait(self):
        foo = HasVariousTraits()
        bar = HasVariousTraits()
        obj = UpdateListener(foo=foo, bar=bar)
        events = []
        with self.assertRaises(ValueError):
            obj.observe(events.append, '*:*')

    def test_anytrait_unobserve(self):
        obj = HasVariousTraits()
        events = []
        obj.observe(events.append, '*')
        obj.foo = 23
        obj.bar = 'on'
        self.assertEqual(len(events), 2)
        obj.observe(events.append, '*', remove=True)
        obj.foo = 232
        obj.bar = 'mid'
        self.assertEqual(len(events), 2)

    def test_property_subclass_observe(self):

        class Base(HasTraits):
            bar = Int()
            foo = Property(Int(), observe='bar')

            def _get_foo(self):
                return self.bar

        class Derived(Base):
            pass
        events = []
        obj = Derived(bar=3)
        obj.observe(events.append, 'foo')
        self.assertEqual(len(events), 0)
        obj.bar = 5
        self.assertEqual(len(events), 1)