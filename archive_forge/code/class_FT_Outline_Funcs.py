from freetype.ft_types import *
class FT_Outline_Funcs(Structure):
    """
    This structure holds a set of callbacks which are called by
    FT_Outline_Decompose.

    move_to: Callback when outline needs to jump to a new path component.

    line_to: Callback to draw a straight line from the current position to
             the control point.

    conic_to: Callback to draw a second-order Bézier curve from the current
              position using the passed control points.

    curve_to: Callback to draw a third-order Bézier curve from the current
              position using the passed control points.

    shift: Passed to FreeType which will transform vectors via
           `x = (x << shift) - delta` and `y = (y << shift) - delta`

    delta: Passed to FreeType which will transform vectors via
           `x = (x << shift) - delta` and `y = (y << shift) - delta`
    """
    _fields_ = [('move_to', FT_Outline_MoveToFunc), ('line_to', FT_Outline_LineToFunc), ('conic_to', FT_Outline_ConicToFunc), ('cubic_to', FT_Outline_CubicToFunc), ('shift', c_int), ('delta', FT_Pos)]