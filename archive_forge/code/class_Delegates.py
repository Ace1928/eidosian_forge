import gc
import time
import unittest
from traits.api import HasTraits, Any, DelegatesTo, Instance, Int
class Delegates(HasTraits):
    """ Object that delegates. """
    b = Instance(Base)
    i = DelegatesTo('b')