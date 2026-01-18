import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
class EntangledStateError(ValueError):
    """Raised when a product state is expected, but an entangled state is provided."""