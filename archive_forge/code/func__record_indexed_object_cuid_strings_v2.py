import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _record_indexed_object_cuid_strings_v2(obj, cuid_str):
    for idx, data in obj.items():
        cuid_strings[data] = cuid_str + _index_repr(idx)