from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from ..util.deprecation import deprecated
from ..util.strings import format_docstring
from ..util.warnings import warn
from .tickers import Ticker
class LogTickFormatter(TickFormatter):
    """ Display tick values from continuous ranges as powers
    of some base.

    Most often useful in conjunction with a ``LogTicker``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    ticker = Nullable(Instance(Ticker), help='\n    The corresponding ``LogTicker``, used to determine the correct\n    base to use. If unset, the formatter will use base 10 as a default.\n    ')
    min_exponent = Int(0, help='\n    Minimum exponent to format in scientific notation. If not zero\n    all ticks in range from base^-min_exponent to base^min_exponent\n    are displayed without exponential notation.\n    ')