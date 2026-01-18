import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _index_from_slice_info(self, slice_info):
    """
        Constructs an index from the slice_info entry in a slice's
        call stack. The index may then be processed just as any
        other slice index, e.g. from a __getitem__ call in a slice's
        call stack.
        """
    fixed, sliced, ellipsis = slice_info
    if ellipsis is None:
        ellipsis = {}
    else:
        ellipsis = {ellipsis: Ellipsis}
    value_map = {}
    value_map.update(fixed)
    value_map.update(sliced)
    value_map.update(ellipsis)
    return tuple((value_map[i] for i in range(len(value_map))))