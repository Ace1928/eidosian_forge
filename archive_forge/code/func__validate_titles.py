from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
@validate('titles')
def _validate_titles(self, proposal):
    return tuple(pad(proposal.value, '', len(self.children)))