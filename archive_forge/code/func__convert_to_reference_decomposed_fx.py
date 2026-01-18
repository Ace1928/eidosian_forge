from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping
def _convert_to_reference_decomposed_fx(graph_module: GraphModule, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    """ Convert a calibrated or trained model to a reference quantized model, with
    decomposed representation for quantized Tensor
    see https://github.com/pytorch/rfcs/blob/master/RFC-0019-Extending-PyTorch-Quantization-to-Custom-Backends.md for more details,
    reference quantized model is a standard representation of a quantized model provided
    by FX Graph Mode Quantization, it can be further lowered to run on the target
    hardware, like accelerators

    Note: this is not public API

    Args:
        * `graph_module` (GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.
            See :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend. See
            :func:`~torch.ao.quantization.quantize_fx.convert_fx` for more details.

    Return:
        A reference quantized model (GraphModule) with operators working with decomposed quantized Tensor

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        reference_quantized_model = _convert_to_reference_decomposed_fx(prepared_model)

    """
    torch._C._log_api_usage_once('quantization_api.quantize_fx._convert_to_reference_decomposed_fx')
    return _convert_fx(graph_module, is_reference=True, convert_custom_config=convert_custom_config, _remove_qconfig=False, qconfig_mapping=qconfig_mapping, backend_config=backend_config, is_decomposed=True)