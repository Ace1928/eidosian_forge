from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def _get_prepare_custom_config(obj: Any, dict_key: str) -> Optional[PrepareCustomConfig]:
    """
            Convert the given object into a PrepareCustomConfig if possible, else throw an exception.
            """
    if isinstance(obj, PrepareCustomConfig) or obj is None:
        return obj
    if isinstance(obj, Dict):
        return PrepareCustomConfig.from_dict(obj)
    raise ValueError(f"""Expected PrepareCustomConfig in prepare_custom_config_dict["{dict_key}"], got '{type(obj)}'""")