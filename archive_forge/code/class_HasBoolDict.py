import unittest
from traits.api import Bool, Dict, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class HasBoolDict(HasTraits):
    foo = Dict(Bool, Int)