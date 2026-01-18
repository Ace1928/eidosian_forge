import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@validate('n_rows', 'n_columns')
def _validate_integer(self, proposal):
    if proposal['value'] > 0:
        return proposal['value']
    raise TraitError('n_rows and n_columns must be positive integer')