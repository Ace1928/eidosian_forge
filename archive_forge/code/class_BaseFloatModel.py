import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class BaseFloatModel(HasTraits):
    value = BaseFloat
    value_or_none = Either(None, BaseFloat)
    float_or_text = Either(Float, Str)