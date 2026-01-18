from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _check_and_get(self, arr_list, ctx):
    if arr_list is not None:
        if ctx is list:
            return arr_list
        if ctx is None:
            if len(arr_list) == 1:
                return arr_list[0]
            else:
                ctx = context.current_context()
        ctx_list = self._ctx_map[ctx.device_typeid & 1]
        if ctx.device_id < len(ctx_list):
            idx = ctx_list[ctx.device_id]
            if idx is not None:
                return arr_list[idx]
        raise RuntimeError("Parameter '%s' was not initialized on context %s. It was only initialized on %s." % (self.name, str(ctx), str(self._ctx_list)))
    if self._deferred_init:
        raise DeferredInitializationError("Parameter '%s' has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters. You can also avoid deferred initialization by specifying in_units, num_features, etc., for network layers." % self.name)
    raise RuntimeError("Parameter '%s' has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks" % self.name)