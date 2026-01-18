import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@observe('top_left', 'bottom_left', 'top_right', 'bottom_right', 'merge')
def _child_changed(self, change):
    self._update_layout()