import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class TraitsHolder(HasTraits):
    a_no = Instance(IAverage, adapt='no')
    a_yes = Instance(IAverage, adapt='yes')
    a_default = Instance(IAverage, adapt='default')
    list_adapted_to = Supports(IList)
    foo_adapted_to = Supports(IFoo)
    foo_plus_adapted_to = Supports(IFooPlus)
    list_adapts_to = AdaptsTo(IList)
    foo_adapts_to = AdaptsTo(IFoo)
    foo_plus_adapts_to = AdaptsTo(IFooPlus)