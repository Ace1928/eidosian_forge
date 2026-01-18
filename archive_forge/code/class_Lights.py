import collections
import enum
import unittest
from traits.trait_base import safe_contains
class Lights(enum.Enum):
    red = 'red'
    blue = 'blue'
    green = 'green'