import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class UndeclaredAverageProvider(HasTraits):
    """
    Class that conforms to the IAverage interface, but doesn't declare
    that it does so.
    """

    def get_average(self):
        return 5.6