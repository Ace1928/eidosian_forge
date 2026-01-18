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
def _summary_hook(block, _, outputs):
    class_name = block.__class__.__name__
    block_idx = len(summary) - 1
    m_key = '%s-%i' % (class_name, block_idx + 1)
    summary[m_key] = OrderedDict()
    summary[m_key]['output_shape'] = _get_shape_str(outputs)
    params = 0
    summary[m_key]['trainable'] = 0
    summary[m_key]['shared'] = 0
    for p in block.params.values():
        params += p.data().size
        summary[m_key]['trainable'] += 0 if p.grad_req == 'null' else p.data().size
        if p in seen:
            summary[m_key]['shared'] += p.data().size
        else:
            seen.add(p)
    summary[m_key]['n_params'] = params