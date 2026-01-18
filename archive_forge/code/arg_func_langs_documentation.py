import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
Extracts an InternalGate object from an InternalGate proto.

    Args:
        msg: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.

    Returns:
        The deserialized InternalGate object.

    Raises:
        ValueError: On failure to parse any of the gate arguments.
    