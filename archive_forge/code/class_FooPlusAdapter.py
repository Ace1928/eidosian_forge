import contextlib
import logging
import unittest
from traits import has_traits
from traits.api import (
from traits.adaptation.api import reset_global_adaptation_manager
from traits.interface_checker import InterfaceError
class FooPlusAdapter(object):

    def __init__(self, obj):
        self.obj = obj

    def get_foo(self):
        return self.obj.get_foo()

    def get_foo_plus(self):
        return self.obj.get_foo() + 1