import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
class TextPoint:
    """Iterable, but values trigger TypeError in Affine.__mul__."""

    def __iter__(self):
        return ('1', '2')

    def __rmul__(self, other):
        return other * (1, 2)