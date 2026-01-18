import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _phased_x_to_proto(gate: cirq.PhasedXPowGate, q: cirq.Qid) -> operations_pb2.ExpW:
    return operations_pb2.ExpW(target=_qubit_to_proto(q), axis_half_turns=_parameterized_value_to_proto(gate.phase_exponent), half_turns=_parameterized_value_to_proto(gate.exponent))