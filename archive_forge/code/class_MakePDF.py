from math import pi
import pytest
from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
from IPython.utils.io import capture_output
class MakePDF(object):

    def _repr_pdf_(self):
        return 'PDF'