from argparse import Namespace
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union
import numpy as np
from torch import Tensor
def _sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

    Args:
        params: Dictionary containing the hyperparameters

    Returns:
        dictionary with all callables sanitized

    """

    def _sanitize_callable(val: Any) -> Any:
        if callable(val):
            try:
                _val = val()
                if callable(_val):
                    return val.__name__
                return _val
            except Exception:
                return getattr(val, '__name__', None)
        return val
    return {key: _sanitize_callable(val) for key, val in params.items()}