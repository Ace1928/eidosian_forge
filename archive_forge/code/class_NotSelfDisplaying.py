from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
class NotSelfDisplaying(object):

    def __repr__(self):
        return 'NotSelfDisplaying'

    def _ipython_display_(self):
        raise NotImplementedError