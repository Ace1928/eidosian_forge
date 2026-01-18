from __future__ import annotations
import uuid
from collections import OrderedDict
from functools import lru_cache, partial
from html import escape
from importlib.resources import files
from xarray.core.formatting import (
from xarray.core.options import _get_boolean_with_default
def _obj_repr(obj, header_components, sections):
    """Return HTML repr of an xarray object.

    If CSS is not injected (untrusted notebook), fallback to the plain text repr.

    """
    header = f"<div class='xr-header'>{''.join((h for h in header_components))}</div>"
    sections = ''.join((f"<li class='xr-section-item'>{s}</li>" for s in sections))
    icons_svg, css_style = _load_static_files()
    return f"<div>{icons_svg}<style>{css_style}</style><pre class='xr-text-repr-fallback'>{escape(repr(obj))}</pre><div class='xr-wrap' style='display:none'>{header}<ul class='xr-sections'>{sections}</ul></div></div>"