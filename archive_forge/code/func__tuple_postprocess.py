from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def _tuple_postprocess(res, to_unpack):
    if isinstance(to_unpack, tuple):
        assert len(to_unpack) == 2
        if not to_unpack[1]:
            res = tuple((el[0] for el in res))
        if not to_unpack[0]:
            res = res[0]
    elif not to_unpack:
        res = res[0]
    return res