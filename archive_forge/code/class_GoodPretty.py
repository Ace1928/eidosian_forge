from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
class GoodPretty(object):

    def _repr_pretty_(self, pp, cycle):
        pp.text('foo')

    def __repr__(self):
        return 'GoodPretty()'