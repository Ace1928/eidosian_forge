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
def _prepare_standalone_module_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], is_qat: bool, example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    """ [Internal use only] Prepare a standalone module, so that it can be used when quantizing the
    parent module.
    standalone_module means it a submodule that is not inlined in parent module,
    and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module

    Returns:

        * model(GraphModule): prepared standalone module. It has these attributes in
          model.meta:

            * `standalone_module_input_quantized_idxs(List[Int])`: a list of
              indexes for the graph input that is expected to be quantized,
              same as input_quantized_idxs configuration provided
              for the standalone module
            * `standalone_module_output_quantized_idxs(List[Int])`: a list of
              indexs for the graph output that is quantized
              same as input_quantized_idxs configuration provided
              for the standalone module

    """
    return _prepare_fx(model, qconfig_mapping, is_qat, example_inputs, prepare_custom_config, backend_config=backend_config, is_standalone_module=True)