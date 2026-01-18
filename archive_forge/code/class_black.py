from __future__ import annotations
import logging # isort:skip
from .util import ColorGroup
class black(ColorGroup):
    """ CSS "Black" Color Group as defined by https://www.w3schools.com/colors/colors_groups.asp

    .. bokeh-color:: gainsboro
    .. bokeh-color:: lightgray
    .. bokeh-color:: silver
    .. bokeh-color:: darkgray
    .. bokeh-color:: gray
    .. bokeh-color:: dimgray
    .. bokeh-color:: lightslategray
    .. bokeh-color:: slategray
    .. bokeh-color:: darkslategray
    .. bokeh-color:: black
    """
    _colors = ('Gainsboro', 'LightGray', 'Silver', 'DarkGray', 'Gray', 'DimGray', 'LightSlateGray', 'SlateGray', 'DarkSlateGray', 'Black')