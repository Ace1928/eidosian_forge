import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def _parallel_gate_op(gate, qubits):
    return cirq.parallel_gate_op(gate, *qubits)