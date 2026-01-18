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
def _get_standalone_module_configs(node: Node, named_modules: Dict[str, torch.nn.Module], prepare_custom_config: PrepareCustomConfig, parent_qconfig: QConfigAny, parent_backend_config: Optional[BackendConfig]) -> Tuple[QConfigMapping, Tuple[Any, ...], PrepareCustomConfig, Optional[BackendConfig]]:
    """
    Returns the standalone module QConfigMapping and PrepareCustomConfig
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    module_name = str(node.target)
    module_type = type(named_modules[module_name])
    config_entry = StandaloneModuleConfigEntry(None, (), None, None)
    config_entry = prepare_custom_config.standalone_module_classes.get(module_type, config_entry)
    config_entry = prepare_custom_config.standalone_module_names.get(module_name, config_entry)
    qconfig_mapping = config_entry.qconfig_mapping or QConfigMapping().set_global(parent_qconfig)
    example_inputs = config_entry.example_inputs
    prepare_custom_config = config_entry.prepare_custom_config or PrepareCustomConfig()
    backend_config = config_entry.backend_config or parent_backend_config
    return (qconfig_mapping, example_inputs, prepare_custom_config, backend_config)