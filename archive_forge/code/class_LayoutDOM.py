from __future__ import annotations
import logging # isort:skip
from ..colors import RGB, Color, ColorLike
from ..core.enums import (
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_aliases import GridSpacing, Pixels, TracksSizing
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import (
from ..model import Model
from .ui.panes import Pane
from .ui.tooltips import Tooltip
from .ui.ui_element import UIElement
@abstract
class LayoutDOM(Pane):
    """ The base class for layoutable components.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    disabled = Bool(False, help='\n    Whether the widget will be disabled when rendered.\n\n    If ``True``, the widget will be greyed-out and not responsive to UI events.\n    ')
    width: int | None = Nullable(NonNegative(Int), help='\n    The width of the component (in pixels).\n\n    This can be either fixed or preferred width, depending on width sizing policy.\n    ')
    height: int | None = Nullable(NonNegative(Int), help='\n    The height of the component (in pixels).\n\n    This can be either fixed or preferred height, depending on height sizing policy.\n    ')
    min_width = Nullable(NonNegative(Int), help='\n    Minimal width of the component (in pixels) if width is adjustable.\n    ')
    min_height = Nullable(NonNegative(Int), help='\n    Minimal height of the component (in pixels) if height is adjustable.\n    ')
    max_width = Nullable(NonNegative(Int), help='\n    Maximal width of the component (in pixels) if width is adjustable.\n    ')
    max_height = Nullable(NonNegative(Int), help='\n    Maximal height of the component (in pixels) if height is adjustable.\n    ')
    margin = Nullable(Either(Int, Tuple(Int, Int), Tuple(Int, Int, Int, Int)), help='\n    Allows to create additional space around the component.\n    The values in the tuple are ordered as follows - Margin-Top, Margin-Right, Margin-Bottom and Margin-Left,\n    similar to CSS standards.\n    Negative margin values may be used to shrink the space from any direction.\n    ')
    width_policy = Either(Auto, Enum(SizingPolicy), default='auto', help='\n    Describes how the component should maintain its width.\n\n    ``"auto"``\n        Use component\'s preferred sizing policy.\n\n    ``"fixed"``\n        Use exactly ``width`` pixels. Component will overflow if it can\'t fit in the\n        available horizontal space.\n\n    ``"fit"``\n        Use component\'s preferred width (if set) and allow it to fit into the available\n        horizontal space within the minimum and maximum width bounds (if set). Component\'s\n        width neither will be aggressively minimized nor maximized.\n\n    ``"min"``\n        Use as little horizontal space as possible, not less than the minimum width (if set).\n        The starting point is the preferred width (if set). The width of the component may\n        shrink or grow depending on the parent layout, aspect management and other factors.\n\n    ``"max"``\n        Use as much horizontal space as possible, not more than the maximum width (if set).\n        The starting point is the preferred width (if set). The width of the component may\n        shrink or grow depending on the parent layout, aspect management and other factors.\n\n    .. note::\n        This is an experimental feature and may change in future. Use it at your\n        own discretion. Prefer using ``sizing_mode`` if this level of control isn\'t\n        strictly necessary.\n\n    ')
    height_policy = Either(Auto, Enum(SizingPolicy), default='auto', help='\n    Describes how the component should maintain its height.\n\n    ``"auto"``\n        Use component\'s preferred sizing policy.\n\n    ``"fixed"``\n        Use exactly ``height`` pixels. Component will overflow if it can\'t fit in the\n        available vertical space.\n\n    ``"fit"``\n        Use component\'s preferred height (if set) and allow to fit into the available\n        vertical space within the minimum and maximum height bounds (if set). Component\'s\n        height neither will be aggressively minimized nor maximized.\n\n    ``"min"``\n        Use as little vertical space as possible, not less than the minimum height (if set).\n        The starting point is the preferred height (if set). The height of the component may\n        shrink or grow depending on the parent layout, aspect management and other factors.\n\n    ``"max"``\n        Use as much vertical space as possible, not more than the maximum height (if set).\n        The starting point is the preferred height (if set). The height of the component may\n        shrink or grow depending on the parent layout, aspect management and other factors.\n\n    .. note::\n        This is an experimental feature and may change in future. Use it at your\n        own discretion. Prefer using ``sizing_mode`` if this level of control isn\'t\n        strictly necessary.\n\n    ')
    aspect_ratio = Either(Null, Auto, Float, help='\n    Describes the proportional relationship between component\'s width and height.\n\n    This works if any of component\'s dimensions are flexible in size. If set to\n    a number, ``width / height = aspect_ratio`` relationship will be maintained.\n    Otherwise, if set to ``"auto"``, component\'s preferred width and height will\n    be used to determine the aspect (if not set, no aspect will be preserved).\n\n    ')
    flow_mode = Enum(FlowMode, default='block', help='\n    Defines whether the layout will flow in the ``block`` or ``inline`` dimension.\n    ')
    sizing_mode = Nullable(Enum(SizingMode), help='\n    How the component should size itself.\n\n    This is a high-level setting for maintaining width and height of the component. To\n    gain more fine grained control over sizing, use ``width_policy``, ``height_policy``\n    and ``aspect_ratio`` instead (those take precedence over ``sizing_mode``).\n\n    Possible scenarios:\n\n    ``"inherit"``\n        The sizing mode is inherited from the parent layout. If there is no parent\n        layout (or parent is not a layout), then this value is treated as if no\n        value for ``sizing_mode`` was provided.\n\n    ``"fixed"``\n        Component is not responsive. It will retain its original width and height\n        regardless of any subsequent browser window resize events.\n\n    ``"stretch_width"``\n        Component will responsively resize to stretch to the available width, without\n        maintaining any aspect ratio. The height of the component depends on the type\n        of the component and may be fixed or fit to component\'s contents.\n\n    ``"stretch_height"``\n        Component will responsively resize to stretch to the available height, without\n        maintaining any aspect ratio. The width of the component depends on the type\n        of the component and may be fixed or fit to component\'s contents.\n\n    ``"stretch_both"``\n        Component is completely responsive, independently in width and height, and\n        will occupy all the available horizontal and vertical space, even if this\n        changes the aspect ratio of the component.\n\n    ``"scale_width"``\n        Component will responsively resize to stretch to the available width, while\n        maintaining the original or provided aspect ratio.\n\n    ``"scale_height"``\n        Component will responsively resize to stretch to the available height, while\n        maintaining the original or provided aspect ratio.\n\n    ``"scale_both"``\n        Component will responsively resize to both the available width and height, while\n        maintaining the original or provided aspect ratio.\n\n    ')
    align = Either(Auto, Enum(Align), Tuple(Enum(Align), Enum(Align)), default='auto', help='\n    The alignment point within the parent container.\n\n    This property is useful only if this component is a child element of a layout\n    (e.g. a grid). Self alignment can be overridden by the parent container (e.g.\n    grid track align).\n    ')
    resizable = Either(Bool, Enum(Dimensions), default=False, help='\n    Whether the layout is interactively resizable, and if so in which dimensions.\n    ')

    def _set_background(self, color: ColorLike) -> None:
        """ Background color of the component. """
        if isinstance(color, Color):
            color = color.to_css()
        elif isinstance(color, tuple):
            color = RGB.from_tuple(color).to_css()
        if isinstance(self.styles, dict):
            self.styles['background-color'] = color
        else:
            self.styles.background_color = color
    background = property(None, _set_background)

    @warning(FIXED_SIZING_MODE)
    def _check_fixed_sizing_mode(self):
        if self.sizing_mode == 'fixed' and (self.width is None or self.height is None):
            return str(self)

    @warning(FIXED_WIDTH_POLICY)
    def _check_fixed_width_policy(self):
        if self.width_policy == 'fixed' and self.width is None:
            return str(self)

    @warning(FIXED_HEIGHT_POLICY)
    def _check_fixed_height_policy(self):
        if self.height_policy == 'fixed' and self.height is None:
            return str(self)

    @error(MIN_PREFERRED_MAX_WIDTH)
    def _check_min_preferred_max_width(self):
        min_width = self.min_width if self.min_width is not None else 0
        width = self.width if self.width is not None and (self.sizing_mode == 'fixed' or self.width_policy == 'fixed') else min_width
        max_width = self.max_width if self.max_width is not None else width
        if not min_width <= width <= max_width:
            return str(self)

    @error(MIN_PREFERRED_MAX_HEIGHT)
    def _check_min_preferred_max_height(self):
        min_height = self.min_height if self.min_height is not None else 0
        height = self.height if self.height is not None and (self.sizing_mode == 'fixed' or self.height_policy == 'fixed') else min_height
        max_height = self.max_height if self.max_height is not None else height
        if not min_height <= height <= max_height:
            return str(self)

    def _sphinx_height_hint(self) -> int | None:
        if self.sizing_mode in ('stretch_width', 'fixed', None):
            return self.height
        return None