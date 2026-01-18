import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
def _conv_forwards(input: List[int], weight: List[int], bias: Optional[List[int]], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool) -> List[int]:
    return conv_forwards(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)