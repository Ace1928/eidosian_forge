import json
from typing import Any, cast, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Iterator
import numpy as np
import sympy
import cirq
from cirq_google.api.v1 import operations_pb2
def _parameterized_value_from_proto(proto: operations_pb2.ParameterizedFloat) -> cirq.TParamVal:
    if proto.HasField('parameter_key'):
        return sympy.Symbol(proto.parameter_key)
    if proto.HasField('raw'):
        return proto.raw
    raise ValueError(f'No value specified for parameterized float. Expected "raw" or "parameter_key" to be set. proto: {proto!r}')