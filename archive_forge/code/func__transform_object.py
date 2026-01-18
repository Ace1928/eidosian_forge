from __future__ import annotations
import re
import sys
from typing import (
import param  # type: ignore
from pyviz_comms import Comm, JupyterComm  # type: ignore
from ..io.resources import CDN_DIST
from ..util import lazy_load
from .base import ModelPane
def _transform_object(self, obj: Any) -> Dict[str, Any]:
    if obj is None:
        obj = ''
    elif hasattr(obj, '_repr_latex_'):
        obj = obj._repr_latex_()
    elif is_sympy_expr(obj):
        import sympy
        obj = '$' + sympy.latex(obj) + '$'
    return dict(object=obj)