from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _get_dtype_config(obj: Any) -> DTypeConfig:
    """
            Convert the given object into a ``DTypeConfig`` if possible, else throw an exception.
            """
    if isinstance(obj, DTypeConfig):
        return obj
    if isinstance(obj, Dict):
        return DTypeConfig.from_dict(obj)
    raise ValueError(f"""Expected a list of DTypeConfigs in backend_pattern_config_dict["{DTYPE_CONFIGS_DICT_KEY}"], got '{type(obj)}'""")