import inspect
from typing import Dict, List, Union
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import registration
def _symbolic_argument_count(func):
    params = []
    signature = inspect.signature(func)
    optional_params = []
    for name, parameter in signature.parameters.items():
        if name in {'_outputs', 'g'}:
            continue
        if parameter.default is parameter.empty:
            optional_params.append(parameter)
        else:
            params.append(str(parameter))
    return params