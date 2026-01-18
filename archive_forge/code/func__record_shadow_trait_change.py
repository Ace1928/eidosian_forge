import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
@on_trait_change('color_')
def _record_shadow_trait_change(self, obj, name, old, new):
    change = (obj, name, old, new)
    self.shadow_changes.append(change)