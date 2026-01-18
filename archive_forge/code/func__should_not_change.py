import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def _should_not_change(self, comp):

    def no_changes(*args):
        self.fail('Nothing should get changed')
    comp.changed = no_changes
    comp.adapters.changed = no_changes
    comp.adapters._v_lookup.changed = no_changes