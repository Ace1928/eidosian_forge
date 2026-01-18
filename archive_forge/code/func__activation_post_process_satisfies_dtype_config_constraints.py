import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def _activation_post_process_satisfies_dtype_config_constraints(activation_post_process: Union[ObserverBase, FakeQuantizeBase], dtype_with_constraints: DTypeWithConstraints, debug_string: str) -> bool:
    observer = _get_observer_from_activation_post_process(activation_post_process)
    app_quant_min = getattr(observer, 'quant_min', None)
    app_quant_max = getattr(observer, 'quant_max', None)
    app_scale_min = getattr(observer, 'eps', None)
    backend_quant_min = dtype_with_constraints.quant_min_lower_bound
    backend_quant_max = dtype_with_constraints.quant_max_upper_bound
    backend_scale_min = dtype_with_constraints.scale_min_lower_bound
    backend_scale_exact_match = dtype_with_constraints.scale_exact_match
    backend_zero_point_exact_match = dtype_with_constraints.zero_point_exact_match
    if backend_quant_min is not None and backend_quant_max is not None:
        if app_quant_min is None or app_quant_max is None:
            warnings.warn(f"QConfig {debug_string} must specify 'quant_min' and 'quant_max', ignoring {qconfig}")
            return False
        elif app_quant_min < backend_quant_min or app_quant_max > backend_quant_max:
            warnings.warn(f"QConfig {debug_string} quantization range must fall within the backend's:\nQConfig range = ({app_quant_min}, {app_quant_max}), BackendConfig range = ({backend_quant_min}, {backend_quant_max}), ignoring {qconfig}")
            return False
    if backend_scale_min is not None:
        if app_scale_min is None:
            warnings.warn(f"QConfig {debug_string} must specify 'eps', ignoring {qconfig}")
            return False
        if app_scale_min < backend_scale_min:
            warnings.warn(f"QConfig {debug_string} eps ({app_scale_min}) must be greater than or equal to the backend's min scale value ({backend_scale_min}), ignoring {qconfig}")
            return False
    if backend_scale_exact_match is not None and backend_zero_point_exact_match is not None:
        for accepted_qconfig in [float16_static_qconfig, float16_dynamic_qconfig]:
            if qconfig_equals(qconfig, accepted_qconfig):
                return True
        suggestion_str = 'Please use torch.ao.quantization.get_default_qconfig_mapping or torch.ao.quantization.get_default_qat_qconfig_mapping. Example:\n    qconfig_mapping = get_default_qconfig_mapping("fbgemm")\n    model = prepare_fx(model, qconfig_mapping, example_inputs)'
        if not isinstance(activation_post_process, FixedQParamsObserver) and (not isinstance(activation_post_process, FixedQParamsFakeQuantize)):
            warnings.warn(f'QConfig must specify a FixedQParamsObserver or a FixedQParamsFakeQuantize for fixed qparams ops, ignoring {qconfig}.\n{suggestion_str}')
            return False
        if observer.scale != backend_scale_exact_match or observer.zero_point != backend_zero_point_exact_match:
            warnings.warn(f"QConfig fixed scale ({observer.scale}) and zero point ({observer.zero_point}) do not match the backend's ({backend_scale_exact_match} and {backend_zero_point_exact_match}), ignoring {qconfig}.\n{suggestion_str}")
            return False
    return True