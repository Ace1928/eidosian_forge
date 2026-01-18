from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _normalize_parse_gpu_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if not isinstance(s, str):
        return s
    if s == '-1':
        return -1
    if ',' in s:
        return [int(x.strip()) for x in s.split(',') if len(x) > 0]
    return int(s.strip())