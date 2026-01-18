import sys
import types
from typing import List
import torch
def _get_qengine_id(qengine: str) -> int:
    if qengine == 'none' or qengine == '' or qengine is None:
        ret = 0
    elif qengine == 'fbgemm':
        ret = 1
    elif qengine == 'qnnpack':
        ret = 2
    elif qengine == 'onednn':
        ret = 3
    elif qengine == 'x86':
        ret = 4
    else:
        ret = -1
        raise RuntimeError(f'{qengine} is not a valid value for quantized engine')
    return ret