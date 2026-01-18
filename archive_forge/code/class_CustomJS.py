from __future__ import annotations
import logging # isort:skip
import pathlib
from typing import TYPE_CHECKING, Any as any
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.bases import Init
from ..core.property.singletons import Intrinsic
from ..core.validation import error
from ..core.validation.errors import INVALID_PROPERTY_VALUE, NOT_A_PROPERTY_OF
from ..model import Model
class CustomJS(CustomCode):
    """ Execute a JavaScript function.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef)(default={}, help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the callback's code snippet as the values of\n    named parameters to the callback.\n    ")
    code = Required(String)(help='\n    A snippet of JavaScript code to execute in the browser.\n\n    This can be interpreted either as a JavaScript function or a module, depending\n    on the ``module`` property:\n\n    1. A JS function.\n\n    The code is made into the body of a function, and all of of the named objects in\n    ``args`` are available as parameters that the code can use. Additionally,\n    a ``cb_obj`` parameter contains the object that triggered the callback\n    an optional ``cb_data`` parameter that contains any tool-specific data\n    (i.e. mouse coordinates and hovered glyph indices for the ``HoverTool``)\n    and additional document context in ``cb_context`` argument.\n\n    2. An ES module.\n\n    A JavaScript module (ESM) exporting a default function with the following\n    signature:\n\n    .. code-block: javascript\n\n        export default function(args, obj, data, context) {\n            // program logic\n        }\n\n    where ``args`` is a key-value mapping of user-provided parameters, ``obj``\n    refers to the object that triggered the callback, ``data`` is a key-value\n    mapping of optional parameters provided by the caller, and ``context`` is\n    an additional document context.\n\n    The additional document context is composed of the following members:\n\n    * ``index``: The view manager governing all views in the current\n      instance of ``Bokeh``. If only one instance of ``Bokeh`` is\n      loaded, then this is equivalent to using ``Bokeh.index``.\n\n    This function can be an asynchronous function (``async function () {}`` or\n    ``async () => {}``) if for example external resources are needed, which\n    would require usage of one of the asynchronous Web APIs, for example:\n\n    .. code-block: javascript\n\n        const response = await fetch("/assets/data.csv")\n        const data = await response.text()\n\n    ')
    module = Either(Auto, Bool, default='auto', help='\n    Whether to interpret the code as a JS function or ES module. If set to\n    ``"auto"``, the this will be inferred from the code.\n    ')

    @classmethod
    def from_file(cls, path: PathLike, **args: any) -> CustomJS:
        """
        Construct a ``CustomJS`` instance from a ``*.js`` or ``*.mjs`` file.

        For example, if we want to construct a ``CustomJS`` instance from
        a JavaScript module ``my_module.mjs``, that takes a single argument
        ``source``, then we would use:

        .. code-block: python

            from bokeh.models import ColumnDataSource, CustomJS
            source = ColumnDataSource(data=dict(x=[1, 2, 3]))
            CustomJS.from_file("./my_module.mjs", source=source)

        """
        path = pathlib.Path(path)
        if path.suffix == '.js':
            module = False
        elif path.suffix == '.mjs':
            module = True
        else:
            raise RuntimeError(f'expected a *.js or *.mjs file, got {path}')
        with open(path, encoding='utf-8') as file:
            code = file.read()
        return CustomJS(code=code, args=args, module=module)