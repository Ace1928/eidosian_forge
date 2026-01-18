from typing import List, Optional, TYPE_CHECKING, Tuple, Sequence
import numpy as np
from cirq import linalg, value
from cirq.sim import simulation_utils
Validates that the indices have values within range of `len(qid_shape)`.