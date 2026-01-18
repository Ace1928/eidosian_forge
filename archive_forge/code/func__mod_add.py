import itertools
from collections import defaultdict
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import linalg, ops, protocols, value
from cirq.linalg import transformations
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.synchronize_terminal_measurements import find_terminal_measurements
def _mod_add(source: 'cirq.Qid', target: 'cirq.Qid') -> 'cirq.Operation':
    assert source.dimension == target.dimension
    if source.dimension == 2:
        return ops.CX(source, target)
    return _ModAdd(source.dimension).on(source, target)