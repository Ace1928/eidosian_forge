import unittest
import warnings
from traits.api import (
class IdentityCompare(HasTraits):
    bar = Any(comparison_mode=ComparisonMode.identity)