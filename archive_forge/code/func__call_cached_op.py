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
def _call_cached_op(self, *args):
    if self._cached_op is None:
        self._build_cache(*args)
    assert self._cached_op, 'Gluon failed to build the cache. This should never happen. Please submit an issue on Github https://github.com/apache/incubator-mxnet.'
    if self._callback:
        self._cached_op._register_op_hook(self._callback, self._monitor_all)
        if len(self._flags) >= 2 and (self._flags[1] or self._flags[0]):
            warnings.warn('register_op_hook is experimental when static_alloc=True / static_shape=True  and may not work correctly')
    args, fmt = _flatten(args, 'input')
    if fmt != self._in_format:
        if len(self._in_format) > len(fmt):
            valid = all([self._in_format[i] == -1 for i in range(len(fmt), len(self._in_format))])
            valid = valid and fmt == self._in_format[:len(fmt)]
        elif len(self._in_format) < len(fmt):
            valid = all([fmt[i] == -1 for i in range(len(self._in_format), len(fmt))])
            valid = valid and fmt[:len(self._in_format)] == self._in_format
        else:
            valid = False
        if not valid:
            raise ValueError('The argument structure of HybridBlock does not match the cached version. Stored format = {}, input format = {}'.format(fmt, self._in_format))
    args_without_none = [ele for ele in args if ele is not None]
    cargs = [args_without_none[i] if is_arg else i.data() for is_arg, i in self._cached_op_args]
    out = self._cached_op(*cargs)
    if isinstance(out, NDArray):
        out = [out]
    return _regroup(out, self._out_format)