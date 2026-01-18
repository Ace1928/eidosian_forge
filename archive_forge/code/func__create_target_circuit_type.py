from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
def _create_target_circuit_type(ops: ops.OP_TREE, target_circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
    return cast(CIRCUIT_TYPE, circuits.Circuit(ops) if isinstance(target_circuit, circuits.Circuit) else circuits.FrozenCircuit(ops))