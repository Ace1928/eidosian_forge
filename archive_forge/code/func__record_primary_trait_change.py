import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
@on_trait_change('color')
def _record_primary_trait_change(self, obj, name, old, new):
    change = (obj, name, old, new)
    self.primary_changes.append(change)