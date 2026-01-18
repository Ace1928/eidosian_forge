import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
def _copy_layout_props(self):
    _props = LayoutProperties.class_trait_names()
    for prop in _props:
        value = getattr(self, prop)
        if value:
            value = self._property_rewrite[prop].get(value, value)
            setattr(self.layout, prop, value)