from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
class BadHTML(object):

    def _repr_html_(self):
        raise ValueError('Bad HTML')