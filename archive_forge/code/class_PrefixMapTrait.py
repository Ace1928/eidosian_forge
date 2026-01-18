import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class PrefixMapTrait(HasTraits):
    value = Trait('one', TraitPrefixMap({'one': 1, 'two': 2, 'three': 3}))