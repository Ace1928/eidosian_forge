import decimal
from numbers import Number
from _plotly_utils import exceptions
from . import (  # noqa: F401
def _constrain_color(c):
    if c > 255.0:
        return 255.0
    elif c < 0.0:
        return 0.0
    else:
        return c