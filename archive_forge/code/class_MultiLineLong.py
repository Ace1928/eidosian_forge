import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
class MultiLineLong:

    def __repr__(self):
        return 'Line 1\nLooooooooooongestLine2\nLongerLine 3'