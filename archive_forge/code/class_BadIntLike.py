import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class BadIntLike(object):
    """
    Object whose __index__ method raises something other than TypeError.
    """

    def __index__(self):
        raise ZeroDivisionError('bogus error')