import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
class CalledAsMethod(HasTraits):
    foo = Float