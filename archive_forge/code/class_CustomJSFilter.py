from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class CustomJSFilter(Filter):
    """ Filter data sources with a custom defined JavaScript function.

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = RestrictedDict(String, AnyRef, disallow=('source',), help="\n    A mapping of names to Python objects. In particular those can be bokeh's models.\n    These objects are made available to the callback's code snippet as the values of\n    named parameters to the callback.\n    ")
    code = String(default='', help="\n    A snippet of JavaScript code to filter data contained in a columnar data source.\n    The code is made into the body of a function, and all of of the named objects in\n    ``args`` are available as parameters that the code can use. The variable\n    ``source`` will contain the data source that is associated with the ``CDSView`` this\n    filter is added to.\n\n    The code should either return the indices of the subset or an array of booleans\n    to use to subset data source rows.\n\n    Example:\n\n    .. code-block::\n\n        code = '''\n        const indices = []\n        for (let i = 0; i <= source.data['some_column'].length; i++) {\n            if (source.data['some_column'][i] == 'some_value') {\n                indices.push(i)\n            }\n        }\n        return indices\n        '''\n\n    ")