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
def _infer_attrs(self, infer_fn, attr, *args):
    """Generic infer attributes."""
    inputs, out = self._get_graph(*args)
    args, _ = _flatten(args, 'input')
    args_without_none = [ele for ele in args if ele is not None]
    with warnings.catch_warnings(record=True) as w:
        arg_attrs, _, aux_attrs = getattr(out, infer_fn)(**{i.name: getattr(j, attr) for i, j in zip(inputs, args_without_none)})
        if arg_attrs is None:
            raise ValueError(w[0].message)
    sdict = {i: j for i, j in zip(out.list_arguments(), arg_attrs)}
    sdict.update({name: attr for name, attr in zip(out.list_auxiliary_states(), aux_attrs)})
    for i in self.collect_params().values():
        setattr(i, attr, sdict[i.name])