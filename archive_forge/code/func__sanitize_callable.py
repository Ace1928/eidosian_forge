from argparse import Namespace
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union
import numpy as np
from torch import Tensor
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