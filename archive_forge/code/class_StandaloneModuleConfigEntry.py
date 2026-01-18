from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
@dataclass
class StandaloneModuleConfigEntry:
    qconfig_mapping: Optional[QConfigMapping]
    example_inputs: Tuple[Any, ...]
    prepare_custom_config: Optional[PrepareCustomConfig]
    backend_config: Optional[BackendConfig]