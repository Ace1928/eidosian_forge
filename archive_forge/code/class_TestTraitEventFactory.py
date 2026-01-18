import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Int
from traits.observation._trait_change_event import (
class TestTraitEventFactory(unittest.TestCase):
    """ Test event factory compatibility with CTrait."""

    def test_trait_change_notification_compat(self):

        class Foo(HasTraits):
            number = Int()
        events = []

        def notifier(*args, **kwargs):
            event = trait_event_factory(*args, **kwargs)
            events.append(event)
        foo = Foo(number=0)
        trait = foo.trait('number')
        trait._notifiers(True).append(notifier)
        foo.number += 1
        event, = events
        self.assertIs(event.object, foo)
        self.assertEqual(event.name, 'number')
        self.assertEqual(event.old, 0)
        self.assertEqual(event.new, 1)