from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
@observe('children')
def _observe_children(self, change):
    self._reset_selected_index()
    self._reset_titles()