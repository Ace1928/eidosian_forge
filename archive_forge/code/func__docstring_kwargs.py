from __future__ import annotations
import logging # isort:skip
from inspect import Parameter
from ..models import Marker
def _docstring_kwargs(parameters):
    arglines = []
    for param, typ, doc in (x for x in parameters if x[0].kind == Parameter.KEYWORD_ONLY):
        _add_arglines(arglines, param, typ, doc)
    return '\n'.join(arglines)