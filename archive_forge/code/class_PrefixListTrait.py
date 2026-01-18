import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class PrefixListTrait(HasTraits):
    value = Trait('one', TraitPrefixList('one', 'two', 'three'))