import inspect
import unittest
from traits.api import (
class MyCallable(HasTraits):
    value = OldCallable()