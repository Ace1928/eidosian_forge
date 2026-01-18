from traitlets import (Bool, Int, Float, Unicode, Dict,
from traittypes import Array
from ipywidgets import Widget, Color, widget_serialization, register
from .scales import Scale
from .traits import Date, array_serialization, _array_equal
from .marks import Lines
from ._version import __frontend_version__
import numpy as np
@register_interaction('bqplot.MultiSelector')
class MultiSelector(BrushIntervalSelector):
    """Multi selector interaction.

    This 1-D selector interaction enables the user to select multiple intervals
    using the mouse. A mouse-down marks the start of the interval. The drag
    after the mouse down in the x-direction selects the extent and a mouse-up
    signifies the end of the interval.

    The current selector is highlighted with a green border and the inactive
    selectors are highlighted with a red border.

    The multi selector has three modes:
        1. default mode: In this mode the interaction behaves exactly as the
                brush selector interaction with the current selector.
        2. add mode: In this mode a new selector can be added by clicking at
                a point and dragging over the interval of interest. Once a new
                selector has been added, the multi selector is back in the
                default mode.
                From the default mode, ctrl+click switches to the add mode.
        3. choose mode: In this mode, any of the existing inactive selectors
                can be set as the active selector. When an inactive selector is
                selected by clicking, the multi selector goes back to the
                default mode.
                From the default mode, shift+click switches to the choose mode.

    A double click at the same point without moving the mouse in the
    x-direction will result in the entire interval being selected for the
    current selector.

    Attributes
    ----------
    selected: dict
        A dictionary with keys being the names of the intervals and values
        being the two element arrays containing the start and end of the
        interval selected by that particular selector in terms of the scale of
        the selector.
        This is a read-only attribute.
        This attribute changes while the selection is being made with the
        MultiSelectorinteraction.
    brushing: bool (default: False)
        A boolean attribute to indicate if the selector is being dragged.
        It is True when the selector is being moved and false when it is not.
        This attribute can be used to trigger computationally intensive code
        which should be run only on the interval selection being completed as
        opposed to code which should be run whenever selected is changing.
    names: list
        A list of strings indicating the keys of the different intervals.
        Default values are 'int1', 'int2', 'int3' and so on.
    show_names: bool (default: True)
        Attribute to indicate if the names of the intervals are to be displayed
        along with the interval.
    """
    names = List().tag(sync=True)
    selected = Dict().tag(sync=True)
    _selected = Dict().tag(sync=True)
    show_names = Bool(True).tag(sync=True)

    def __init__(self, **kwargs):
        try:
            self.read_json = kwargs.get('scale').domain_class.from_json
        except AttributeError:
            self.read_json = None
        super(MultiSelector, self).__init__(**kwargs)
        self.on_trait_change(self.hidden_selected_changed, '_selected')

    def hidden_selected_changed(self, name, selected):
        actual_selected = {}
        if self.read_json is None:
            self.selected = self._selected
        else:
            for key in self._selected:
                actual_selected[key] = [self.read_json(elem) for elem in self._selected[key]]
            self.selected = actual_selected
    _view_name = Unicode('MultiSelector').tag(sync=True)
    _model_name = Unicode('MultiSelectorModel').tag(sync=True)