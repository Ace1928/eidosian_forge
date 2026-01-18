import importlib
import inspect
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration
@symbolic_helper.parse_args('v', 't', 't', 't', 't', 't', 't', 't')
def _empty_affine_quantized(g: jit_utils.GraphContext, input, shape, scale, zero_point, dtype, pin_memory, memory_format, layout):
    return input