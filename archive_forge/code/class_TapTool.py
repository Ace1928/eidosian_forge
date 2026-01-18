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
class TapTool(Tap, SelectTool):
    """ *toolbar icon*: |tap_icon|

    The tap selection tool allows the user to select at single points by
    left-clicking a mouse, or tapping with a finger.

    See :ref:`ug_styling_plots_selected_unselected_glyphs` for information
    on styling selected and unselected glyphs.

    .. |tap_icon| image:: /_images/icons/Tap.png
        :height: 24px
        :alt:  Icon of two concentric circles with a + in the lower right representing the tap tool in the toolbar.

    .. note::
        Selections can be comprised of multiple regions, even those
        made by different selection tools. Hold down the SHIFT key
        while making a selection to append the new selection to any
        previous selection that might exist.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    behavior = Enum('select', 'inspect', default='select', help="\n    This tool can be configured to either make selections or inspections\n    on associated data sources. The difference is that selection changes\n    propagate across bokeh and other components (e.g. selection glyph)\n    will be notified. Inspections don't act like this, so it's useful to\n    configure `callback` when setting `behavior='inspect'`.\n    ")
    gesture = Enum('tap', 'doubletap', default='tap', help='\n    Specifies which kind of gesture will be used to trigger the tool,\n    either a single or double tap.\n    ')
    modifiers = Struct(shift=Optional(Bool), ctrl=Optional(Bool), alt=Optional(Bool), default={}, help='\n    Allows to configure a combination of modifier keys, which need to\n    be pressed during the selected gesture for this tool to trigger.\n\n    .. warning::\n        Configuring modifiers is a platform dependent feature and\n        can make this tool unusable for example on mobile devices.\n\n    ').accepts(Enum(KeyModifier), lambda key_mod: {key_mod: True})
    callback = Nullable(Instance(Callback), help='\n    A callback to execute *whenever a glyph is "hit"* by a mouse click\n    or tap.\n\n    This is often useful with the  :class:`~bokeh.models.callbacks.OpenURL`\n    model to open URLs based on a user clicking or tapping a specific glyph.\n\n    However, it may also be a :class:`~bokeh.models.callbacks.CustomJS`\n    which can execute arbitrary JavaScript code in response to clicking or\n    tapping glyphs. The callback will be executed for each individual glyph\n    that is it hit by a click or tap, and will receive the ``TapTool`` model\n    as  ``cb_obj``. The optional ``cb_data`` will have the data source as\n    its ``.source`` attribute and the selection geometry as its\n    ``.geometries`` attribute.\n\n    The ``.geometries`` attribute has 5 members.\n    ``.type`` is the geometry type, which always a ``.point`` for a tap event.\n    ``.sx`` and ``.sy`` are the screen X and Y coordinates where the tap occurred.\n    ``.x`` and ``.y`` are the converted data coordinates for the item that has\n    been selected. The ``.x`` and ``.y`` values are based on the axis assigned\n    to that glyph.\n\n    .. note::\n        This callback does *not* execute on every tap, only when a glyph is\n        "hit". If you would like to execute a callback on every mouse tap,\n        please see :ref:`ug_interaction_js_callbacks_customjs_js_on_event`.\n\n    ')
    mode = Override(default='xor')