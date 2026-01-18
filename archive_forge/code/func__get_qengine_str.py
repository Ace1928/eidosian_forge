import sys
import types
from typing import List
import torch
def _get_qengine_str(qengine: int) -> str:
    all_engines = {0: 'none', 1: 'fbgemm', 2: 'qnnpack', 3: 'onednn', 4: 'x86'}
    return all_engines.get(qengine, '*undefined')