import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class IFooPlus(IFoo):

    def get_foo_plus(self):
        """ Returns even more foo. """