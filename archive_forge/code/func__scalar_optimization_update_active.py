import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _scalar_optimization_update_active(work, res, res_work_pairs, active, mask=None):
    update_dict = {key1: work[key2] for key1, key2 in res_work_pairs}
    update_dict['success'] = work.status == 0
    if mask is not None:
        active_mask = active[mask]
        for key, val in update_dict.items():
            res[key][active_mask] = val[mask] if np.size(val) > 1 else val
    else:
        for key, val in update_dict.items():
            res[key][active] = val