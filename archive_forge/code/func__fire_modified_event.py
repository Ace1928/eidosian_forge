import unittest
from unittest import mock
from traits.trait_types import Any, Dict, Event, Str, TraitDictObject
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_errors import TraitError
@on_trait_change('foos_items,foos.modified')
def _fire_modified_event(self, obj, trait_name, old, new):
    self.modified = True