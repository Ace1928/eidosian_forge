import unittest
from traits.api import BaseInstance, HasStrictTraits, Instance, TraitError
class HasSlices(HasStrictTraits):
    my_slice = Instance(slice)
    also_my_slice = Slice()
    none_explicitly_allowed = Instance(slice, allow_none=True)
    also_allow_none = Slice(allow_none=True)
    disallow_none = Instance(slice, allow_none=False)
    also_disallow_none = Slice(allow_none=False)