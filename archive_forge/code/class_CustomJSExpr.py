from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class CustomJSExpr(Expression):
    """ Evaluate a JavaScript function/generator.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the callback's code snippet as the values of\n    named parameters to the callback. There is no need to manually include the data\n    source of the associated glyph renderer, as it is available within the scope of\n    the code via `this` keyword (e.g. `this.data` will give access to raw data).\n    ")
    code = String(default='', help='\n    A snippet of JavaScript code to execute in the browser. The code is made into\n    the body of a generator function, and all of of the named objects in ``args``\n    are available as parameters that the code can use. One can either return an\n    array-like object (array, typed array, nd-array), an iterable (which will\n    be converted to an array) or a scalar value (which will be converted into\n    an array of an appropriate length), or alternatively yield values that will\n    be collected into an array.\n    ')