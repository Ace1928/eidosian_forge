import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class list_value(HasTraits):
    list1 = Trait([2], TraitList(Trait([1, 2, 3, 4]), maxlen=4))
    list2 = Trait([2], TraitList(Trait([1, 2, 3, 4]), minlen=1, maxlen=4))
    alist = List()