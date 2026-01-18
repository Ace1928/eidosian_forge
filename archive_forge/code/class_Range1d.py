from __future__ import annotations
import logging # isort:skip
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
class Range1d(NumericalRange):
    """ A fixed, closed range [start, end] in a continuous scalar
    dimension.

    In addition to supplying ``start`` and ``end`` keyword arguments
    to the ``Range1d`` initializer, you can also instantiate with
    the convenience syntax::

        Range(0, 10) # equivalent to Range(start=0, end=10)

    """

    def __init__(self, *args, **kwargs) -> None:
        if args and ('start' in kwargs or 'end' in kwargs):
            raise ValueError("'start' and 'end' keywords cannot be used with positional arguments")
        if args and len(args) != 2:
            raise ValueError('Only Range1d(start, end) acceptable when using positional arguments')
        if args:
            kwargs['start'] = args[0]
            kwargs['end'] = args[1]
        super().__init__(**kwargs)
    reset_start = Either(Null, Float, Datetime, TimeDelta, help='\n    The start of the range to apply after reset. If set to ``None`` defaults\n    to the ``start`` value during initialization.\n    ')
    reset_end = Either(Null, Float, Datetime, TimeDelta, help='\n    The end of the range to apply when resetting. If set to ``None`` defaults\n    to the ``end`` value during initialization.\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=True), help="\n    The bounds that the range is allowed to go to. Typically used to prevent\n    the user from panning/zooming/etc away from the data.\n\n    If set to ``'auto'``, the bounds will be computed to the start and end of the Range.\n\n    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether your range is\n    increasing or decreasing, the first item should be the minimum value of the range and the\n    second item should be the maximum. Setting min > max will result in a ``ValueError``.\n\n    By default, bounds are ``None`` and your plot to pan/zoom as far as you want. If you only\n    want to constrain one end of the plot, you can set min or max to None.\n\n    Examples:\n\n    .. code-block:: python\n\n        Range1d(0, 1, bounds='auto')  # Auto-bounded to 0 and 1 (Default behavior)\n        Range1d(start=0, end=1, bounds=(0, None))  # Maximum is unbounded, minimum bounded to 0\n\n    ")
    min_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval. If set to ``None`` (default), the minimum\n    interval is not bound. Can be a ``TimeDelta``. ')
    max_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval. Can be a ``TimeDelta``. Note that ``bounds`` can\n    impose an implicit constraint on the maximum interval as well. ')
    start = Override(default=0)
    end = Override(default=1)