from dataclasses import dataclass
from typing import List, Optional, Sequence, TYPE_CHECKING, Dict, Any
import numpy as np
import pandas as pd
from cirq import sim, value
@dataclass(frozen=True)
class _Simulate2qXEBTask:
    """Helper container for executing simulation tasks, potentially via multiprocessing."""
    circuit_i: int
    cycle_depths: Sequence[int]
    circuit: 'cirq.Circuit'
    param_resolver: 'cirq.ParamResolverOrSimilarType'