import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v['dtype'])
        arr = torch.frombuffer(v['data'], dtype=dtype).reshape(v['shape'])
        if sys.byteorder == 'big':
            arr = torch.from_numpy(arr.numpy().byteswap(inplace=False))
        result[k] = arr
    return result