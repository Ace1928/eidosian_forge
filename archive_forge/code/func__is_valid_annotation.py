import operator
import types
import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import QuantizationAnnotation
def _is_valid_annotation(annotation: QuantizationAnnotation) -> bool:
    if annotation is None:
        return False
    input_qspec_map = annotation.input_qspec_map
    output_qspec = annotation.output_qspec
    if len(input_qspec_map) == 0 and output_qspec is None:
        return False
    return True