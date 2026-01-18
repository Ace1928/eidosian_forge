import unittest
from traits.api import HasTraits, Str, Instance, Any
class TestEventOrder(unittest.TestCase):
    """ Tests that demonstrate that trait events are delivered in LIFO
    order rather than FIFO order.

    Baz receives the "effect" event before it receives the "cause" event.
    """

    def test_lifo_order(self):
        foo = Foo(cause='ORIGINAL')
        bar = Bar(foo=foo, test=self)
        baz = Baz(bar=bar, test=self)
        self.events_delivered = []
        foo.cause = 'CHANGE'
        lifo = ['Bar._caused_changed', 'Baz._effect_changed', 'Baz._caused_changed']
        self.assertEqual(self.events_delivered, lifo)