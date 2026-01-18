from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.BrushSelector')
class BrushSelector(TwoDSelector):
    """Brush interval selector interaction.

    This 2-D selector interaction enables the user to select a rectangular
    region using the brushing action of the mouse. A mouse-down marks the
    starting point of the interval. The drag after the mouse down selects the
    rectangle of interest and a mouse-up signifies the end point of
    the interval.

    Once an interval is drawn, the selector can be moved to a new interval by
    dragging the selector to the new interval.

    A double click at the same point without moving the mouse will result in
    the entire interval being selected.

    Attributes
    ----------
    selected_x: numpy.ndarray
        Two element array containing the start and end of the interval selected
        in terms of the x_scale of the selector.
        This attribute changes while the selection is being made with the
        ``BrushSelector``.
    selected_y: numpy.ndarray
        Two element array containing the start and end of the interval selected
        in terms of the y_scale of the selector.
        This attribute changes while the selection is being made with the
        ``BrushSelector``.
    selected: numpy.ndarray
        A 2x2 array containing the coordinates ::

            [[selected_x[0], selected_y[0]],
             [selected_x[1], selected_y[1]]]
    brushing: bool (default: False)
        boolean attribute to indicate if the selector is being dragged.
        It is True when the selector is being moved and False when it is not.
        This attribute can be used to trigger computationally intensive code
        which should be run only on the interval selection being completed as
        opposed to code which should be run whenever selected is changing.
    color: Color or None (default: None)
        Color of the rectangle representing the brush selector.
    """
    clear = Bool().tag(sync=True)
    brushing = Bool().tag(sync=True)
    selected_x = Array(None, allow_none=True).tag(sync=True, **array_serialization)
    selected_y = Array(None, allow_none=True).tag(sync=True, **array_serialization)
    selected = Array(None, allow_none=True)
    color = Color(None, allow_none=True).tag(sync=True)

    @observe('selected_x', 'selected_y')
    def _set_selected(self, change):
        if self.selected_x is None or len(self.selected_x) == 0 or self.selected_y is None or (len(self.selected_y) == 0):
            self.selected = None
        else:
            self.selected = np.array([[self.selected_x[0], self.selected_y[0]], [self.selected_x[1], self.selected_y[1]]])

    @observe('selected')
    def _set_selected_xy(self, change):
        value = self.selected
        if self.selected is None or len(self.selected) == 0:
            if not (self.selected_x is None or len(self.selected_x) == 0 or self.selected_y is None or (len(self.selected_y) == 0)):
                self.selected_x = None
                self.selected_y = None
        else:
            (x0, y0), (x1, y1) = value
            x = [x0, x1]
            y = [y0, y1]
            with self.hold_sync():
                if not _array_equal(self.selected_x, x):
                    self.selected_x = x
                if not _array_equal(self.selected_y, y):
                    self.selected_y = y
    _view_name = Unicode('BrushSelector').tag(sync=True)
    _model_name = Unicode('BrushSelectorModel').tag(sync=True)