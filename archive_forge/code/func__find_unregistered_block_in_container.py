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
def _find_unregistered_block_in_container(data):
    if isinstance(data, (list, tuple)):
        for ele in data:
            if _find_unregistered_block_in_container(ele):
                return True
        return False
    elif isinstance(data, dict):
        for _, v in data.items():
            if _find_unregistered_block_in_container(v):
                return True
        return False
    elif isinstance(data, Block):
        return not data in children
    else:
        return False