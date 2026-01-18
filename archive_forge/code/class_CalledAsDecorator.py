import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
class CalledAsDecorator(HasTraits):
    foo = Float
    callback = Callable

    @on_trait_change('foo', dispatch='ui')
    def on_foo_change(self, obj, name, old, new):
        self.callback(obj, name, old, new)