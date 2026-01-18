import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _load_json_bool(b: Any):
    """Converts a json field to bool.  If already a bool, pass through."""
    if isinstance(b, bool):
        return b
    return json.loads(b)