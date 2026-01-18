from typing import Tuple, cast
from cirq import circuits, ops, protocols, transformers
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
Returns the number of operations removed at or before start_i.