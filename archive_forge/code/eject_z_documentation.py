from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np
from cirq import ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
Zeroes qubit_phase entries by emitting Z gates.