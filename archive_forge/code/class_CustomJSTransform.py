from __future__ import annotations
import logging # isort:skip
from ..core.enums import JitterRandomDistribution, StepMode
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .sources import ColumnarDataSource
class CustomJSTransform(Transform):
    """ Apply a custom defined transform to data.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the transform' code snippet as the values of\n    named parameters to the callback.\n    ")
    func = String(default='', help="\n    A snippet of JavaScript code to transform a single value. The variable\n    ``x`` will contain the untransformed value and can be expected to be\n    present in the function namespace at render time. The snippet will be\n    into the body of a function and therefore requires a return statement.\n\n    **Example**\n\n    .. code-block:: javascript\n\n        func = '''\n        return Math.floor(x) + 0.5\n        '''\n    ")
    v_func = String(default='', help="\n    A snippet of JavaScript code to transform an array of values. The variable\n    ``xs`` will contain the untransformed array and can be expected to be\n    present in the function namespace at render time. The snippet will be\n    into the body of a function and therefore requires a return statement.\n\n    **Example**\n\n    .. code-block:: javascript\n\n        v_func = '''\n        const new_xs = new Array(xs.length)\n        for(let i = 0; i < xs.length; i++) {\n            new_xs[i] = xs[i] + 0.5\n        }\n        return new_xs\n        '''\n\n    .. warning::\n        The vectorized function, ``v_func``, must return an array of the\n        same length as the input ``xs`` array.\n    ")