import codecs
import re
import ply.lex
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import pickle
from pyomo.common.deprecation import deprecated
from pyomo.core.base.component_namer import (
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import Reference
def _record_indexed_object_cuid_strings_v1(obj, cuid_str):
    _unknown = lambda x: '?' + str(x)
    for idx, data in obj.items():
        if idx.__class__ is tuple and len(idx) > 1:
            cuid_strings[data] = cuid_str + ':' + ','.join((ComponentUID._repr_v1_map.get(x.__class__, _unknown)(x) for x in idx))
        else:
            cuid_strings[data] = cuid_str + ':' + ComponentUID._repr_v1_map.get(idx.__class__, _unknown)(idx)