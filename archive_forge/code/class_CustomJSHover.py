from __future__ import annotations
import logging # isort:skip
import difflib
import typing as tp
from math import nan
from typing import Literal
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.validation import error
from ..core.validation.errors import NO_RANGE_TOOL_RANGES
from ..model import Model
from ..util.strings import nice_join
from .annotations import BoxAnnotation, PolyAnnotation, Span
from .callbacks import Callback
from .dom import Template
from .glyphs import (
from .nodes import Node
from .ranges import Range
from .renderers import DataRenderer, GlyphRenderer
from .ui import UIElement
class CustomJSHover(Model):
    ''' Define a custom formatter to apply to a hover tool field.

    This model can be configured with JavaScript code to format hover tooltips.
    The JavaScript code has access to the current value to format, some special
    variables, and any format configured on the tooltip. The variable ``value``
    will contain the untransformed value. The variable ``special_vars`` will
    provide a dict with the following contents:

    * ``x`` data-space x-coordinate of the mouse
    * ``y`` data-space y-coordinate of the mouse
    * ``sx`` screen-space x-coordinate of the mouse
    * ``sy`` screen-space y-coordinate of the mouse
    * ``data_x`` data-space x-coordinate of the hovered glyph
    * ``data_y`` data-space y-coordinate of the hovered glyph
    * ``indices`` column indices of all currently hovered glyphs
    * ``name`` value of the ``name`` property of the hovered glyph renderer

    If the hover is over a "multi" glyph such as ``Patches`` or ``MultiLine``
    then a ``segment_index`` key will also be present.

    Finally, the value of the format passed in the tooltip specification is
    available as the ``format`` variable.

    Example:

        As an example, the following code adds a custom formatter to format
        WebMercator northing coordinates (in meters) as a latitude:

        .. code-block:: python

            lat_custom = CustomJSHover(code="""
                const projections = Bokeh.require("core/util/projections");
                const x = special_vars.x
                const y = special_vars.y
                const coords = projections.wgs84_mercator.invert(x, y)
                return "" + coords[1]
            """)

            p.add_tools(HoverTool(
                tooltips=[( 'lat','@y{custom}' )],
                formatters={'@y':lat_custom}
            ))

    .. warning::
        The explicit purpose of this Bokeh Model is to embed *raw JavaScript
        code* for a browser to execute. If any part of the code is derived
        from untrusted user inputs, then you must take appropriate care to
        sanitize the user input prior to passing to Bokeh.

    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    args = Dict(String, AnyRef, help='\n    A mapping of names to Bokeh plot objects. These objects are made available\n    to the callback code snippet as the values of named parameters to the\n    callback.\n    ')
    code = String(default='', help='\n    A snippet of JavaScript code to transform a single value. The variable\n    ``value`` will contain the untransformed value and can be expected to be\n    present in the function namespace at render time. Additionally, the\n    variable ``special_vars`` will be available, and will provide a dict\n    with the following contents:\n\n    * ``x`` data-space x-coordinate of the mouse\n    * ``y`` data-space y-coordinate of the mouse\n    * ``sx`` screen-space x-coordinate of the mouse\n    * ``sy`` screen-space y-coordinate of the mouse\n    * ``data_x`` data-space x-coordinate of the hovered glyph\n    * ``data_y`` data-space y-coordinate of the hovered glyph\n    * ``indices`` column indices of all currently hovered glyphs\n\n    If the hover is over a "multi" glyph such as ``Patches`` or ``MultiLine``\n    then a ``segment_index`` key will also be present.\n\n    Finally, the value of the format passed in the tooltip specification is\n    available as the ``format`` variable.\n\n    The snippet will be made into the body of a function and therefore requires\n    a return statement.\n\n    **Example**\n\n    .. code-block:: javascript\n\n        code = \'\'\'\n        return value + " total"\n        \'\'\'\n    ')