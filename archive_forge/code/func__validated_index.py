from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
@validate('selected_index')
def _validated_index(self, proposal):
    if proposal.value is None or 0 <= proposal.value < len(self.children):
        return proposal.value
    else:
        raise TraitError('Invalid selection: index out of bounds')