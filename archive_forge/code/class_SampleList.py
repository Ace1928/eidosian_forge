import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
@provides(IList)
class SampleList(HasTraits):
    """SampleList docstring."""
    data = List(Int, [10, 20, 30])

    def get_list(self):
        return self.data