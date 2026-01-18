import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
class TestMethodUINotifiers(BaseTestUINotifiers, unittest.TestCase):
    """ Tests for dynamic notifiers with `dispatch='ui'` set by method call.
    """

    def obj_factory(self):
        obj = CalledAsMethod()
        obj.on_trait_change(self.on_foo_notifications, 'foo', dispatch='ui')
        return obj