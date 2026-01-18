import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, on_trait_change
class FuzException(Exception):
    pass