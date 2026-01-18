import copy
import torch
import warnings
from torch.fx import (
from torch.fx.graph import (
from torch.fx.node import Argument
from ..quantize import (
from ..observer import (
from ..qconfig import (
from ..qconfig_mapping import (
from .qconfig_mapping_utils import (
from .quantize_handler import (
from torch.ao.quantization import (
from torch.ao.quantization.utils import (
from ._equalize import (
from .pattern_utils import (
from .match_utils import (
from .utils import (
from torch.ao.quantization import (
from torch.ao.quantization.quantize import (
from ..utils import (
from ..backend_config.utils import (
from ..backend_config import (
from .custom_config import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
from torch._subclasses import FakeTensor
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
from dataclasses import asdict
def _get_target_activation_dtype_for_node(node: Node, qconfig: QConfigAny, qhandler: Optional[QuantizeHandler], named_modules: Dict[str, torch.nn.Module], backend_config: BackendConfig, cache_for_no_tensor_check: Dict[Node, bool]) -> Dict[str, Any]:
    """
    For each op attribute in the op's input activation, output activation,
    weight, bias - returns the settings of dtype and is_dynamic we expect
    for the `quantize` call in the reference model representation, or None
    if there is no `quantize` call needed.

    For example, if we have a node corresponding to `op0` in

      x0 -> op0 -> x1

    And we want a reference quantized representation to be

      x0 -> quant_static -> dequant -> op0 -> quant_dynamic -> dequant -> x1

    Then this function will return

      {
        "input_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
        "output_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
      }

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    args_have_no_tensors = all_node_args_have_no_tensors(node, named_modules, cache_for_no_tensor_check)
    if args_have_no_tensors:
        return {'input_act_obs_or_fq_ctr': None, 'output_act_obs_or_fq_ctr': None}
    if qconfig is not None:
        act_dtype, weight_dtype, input_act_is_dynamic = get_qconfig_dtypes(qconfig)
        output_act_dtype = act_dtype if not input_act_is_dynamic else torch.float
        bias_dtype = torch.float16 if act_dtype == torch.float16 and weight_dtype == torch.float16 and (not input_act_is_dynamic) else torch.float
        is_general_tensor_value_op = qhandler is not None and qhandler.is_general_tensor_value_op()
        _is_standalone_module = qhandler is not None and qhandler.is_standalone_module()
        weight_index = None
        if isinstance(node, Node) and node.op == 'call_function' and (node.target in backend_config._pattern_complex_format_to_config):
            weight_index = backend_config._pattern_complex_format_to_config[node.target]._input_type_to_index.get('weight')
        bias_index = None
        if isinstance(node, Node) and node.op == 'call_function' and (node.target in backend_config._pattern_complex_format_to_config):
            bias_index = backend_config._pattern_complex_format_to_config[node.target]._input_type_to_index.get('bias')
        return {'input_act_obs_or_fq_ctr': qconfig.activation, 'weight_obs_or_fq_ctr': qconfig.weight, 'bias_obs_or_fq_ctr': PlaceholderObserver.with_args(dtype=bias_dtype), 'weight_index': weight_index, 'bias_index': bias_index, 'output_act_obs_or_fq_ctr': qconfig.activation, 'reuse_input_obs_or_fq': _is_reuse_input_qconfig(qconfig), 'input_output_share_observers': is_general_tensor_value_op, '_is_standalone_module': _is_standalone_module}
    return copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)