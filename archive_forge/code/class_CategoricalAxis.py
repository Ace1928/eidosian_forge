from __future__ import annotations
import logging # isort:skip
from ..core.enums import Align, LabelOrientation
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_mixins import ScalarFillProps, ScalarLineProps, ScalarTextProps
from .formatters import (
from .labeling import AllLabels, LabelingPolicy
from .renderers import GuideRenderer
from .tickers import (
class CategoricalAxis(Axis):
    """ An axis that displays ticks and labels for categorical ranges.

    The ``CategoricalAxis`` can handle factor ranges with up to two levels of
    nesting, including drawing a separator line between top-level groups of
    factors.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(CategoricalTicker))
    formatter = Override(default=InstanceDefault(CategoricalTickFormatter))
    separator_props = Include(ScalarLineProps, prefix='separator', help='\n    The {prop} of the separator line between top-level categorical groups.\n\n    This property always applies to factors in the outermost level of nesting.\n    ')
    separator_line_color = Override(default='lightgrey')
    separator_line_width = Override(default=2)
    group_props = Include(ScalarTextProps, prefix='group', help='\n    The {prop} of the group categorical labels.\n\n    This property always applies to factors in the outermost level of nesting.\n    If the list of categorical factors is flat (i.e. no nesting) then this\n    property has no effect.\n    ')
    group_label_orientation = Either(Enum(LabelOrientation), Float, default='parallel', help='\n    What direction the group label text should be oriented.\n\n    If a number is supplied, the angle of the text is measured from horizontal.\n\n    This property always applies to factors in the outermost level of nesting.\n    If the list of categorical factors is flat (i.e. no nesting) then this\n    property has no effect.\n    ')
    group_text_font_size = Override(default='11px')
    group_text_font_style = Override(default='bold')
    group_text_color = Override(default='grey')
    subgroup_props = Include(ScalarTextProps, prefix='subgroup', help='\n    The {prop} of the subgroup categorical labels.\n\n    This property always applies to factors in the middle level of nesting.\n    If the list of categorical factors is has only zero or one levels of nesting,\n    then this property has no effect.\n    ')
    subgroup_label_orientation = Either(Enum(LabelOrientation), Float, default='parallel', help='\n    What direction the subgroup label text should be oriented.\n\n    If a number is supplied, the angle of the text is measured from horizontal.\n\n    This property always applies to factors in the middle level of nesting.\n    If the list of categorical factors is has only zero or one levels of nesting,\n    then this property has no effect.\n    ')
    subgroup_text_font_size = Override(default='11px')
    subgroup_text_font_style = Override(default='bold')