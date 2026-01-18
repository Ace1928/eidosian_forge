import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
def _convert_sizes(self, size_list):
    return list(map(self._size_to_css, size_list))