import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _qubit_from_proto(proto: operations_pb2.Qubit):
    return cirq.GridQubit(row=proto.row, col=proto.col)