import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def _save_cached_graphs(blk, index, structure):
    mdl = {'orig_name': blk.name}
    name = type(blk).__name__.lower()
    structure[name + str(index[0])] = mdl
    if isinstance(blk, HybridBlock):
        if blk._cached_graph:
            mdl['in_format'] = blk._in_format
            mdl['out_format'] = blk._out_format
            syms, out = blk._cached_graph
            mdl_syms = []
            for sym in syms:
                mdl_syms.append(sym.tojson())
            mdl['inputs'] = mdl_syms
            mdl['symbol'] = out.tojson()
            mdl['hybridized'] = True
        else:
            mdl['hybridized'] = False
    children = dict()
    mdl['children'] = children
    for ch_name, child in blk._children.items():
        index[0] += 1
        children[child.name] = ch_name
        _save_cached_graphs(child, index, mdl)