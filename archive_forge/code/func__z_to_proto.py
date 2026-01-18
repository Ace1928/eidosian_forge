import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _z_to_proto(gate: cirq.ZPowGate, q: cirq.Qid) -> operations_pb2.ExpZ:
    return operations_pb2.ExpZ(target=_qubit_to_proto(q), half_turns=_parameterized_value_to_proto(gate.exponent))