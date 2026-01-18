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
def _swap_ff_with_fxff(model: torch.nn.Module) -> None:
    """ Swap FloatFunctional with FXFloatFunctional
    """
    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
            modules_to_swap.append(name)
        else:
            _swap_ff_with_fxff(module)
    for name in modules_to_swap:
        del model._modules[name]
        model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()