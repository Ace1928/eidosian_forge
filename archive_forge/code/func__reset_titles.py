from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
def _reset_titles(self):
    if len(self.titles) != len(self.children):
        self.titles = tuple(self.titles)