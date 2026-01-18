import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class SampleListAdapter(Adapter):

    def get_list(self):
        obj = self.adaptee
        return [getattr(obj, name) for name in obj.trait_names(sample=True)]