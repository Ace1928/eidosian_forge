import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class Impossible(object):
    """
    Type that never gets instantiated.
    """

    def __init__(self):
        raise TypeError('Cannot instantiate this class')