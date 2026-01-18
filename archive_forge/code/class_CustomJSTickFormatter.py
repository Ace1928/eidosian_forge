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
class CustomJSTickFormatter(TickFormatter):
    """ Display tick values that are formatted by a user-defined function.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the formatter's code snippet as the values of\n    named parameters to the callback.\n    ")
    code = String(default='', help='\n    A snippet of JavaScript code that reformats a single tick to the desired\n    format. The variable ``tick`` will contain the unformatted tick value and\n    can be expected to be present in the code snippet namespace at render time.\n\n    Additionally available variables are:\n\n    * ``ticks``, an array of all axis ticks as positioned by the ticker,\n    * ``index``, the position of ``tick`` within ``ticks``, and\n    * the keys of ``args`` mapping, if any.\n\n    Finding yourself needing to cache an expensive ``ticks``-dependent\n    computation, you can store it on the ``this`` variable.\n\n    **Example**\n\n    .. code-block:: javascript\n\n        code = \'\'\'\n        this.precision = this.precision || (ticks.length > 5 ? 1 : 2);\n        return Math.floor(tick) + " + " + (tick % 1).toFixed(this.precision);\n        \'\'\'\n    ')