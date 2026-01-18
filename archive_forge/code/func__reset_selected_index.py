from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
def _reset_selected_index(self):
    num_children = len(self.children)
    if num_children == 0:
        self.selected_index = None
    elif self.selected_index == None:
        self.selected_index = 0
    elif num_children < self.selected_index:
        self.selected_index = num_children - 1