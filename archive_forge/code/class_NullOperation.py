from typing import Dict, List
import pytest
import numpy as np
import sympy
from google.protobuf import json_format
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
class NullOperation(cirq.Operation):

    @property
    def qubits(self):
        return tuple()

    def with_qubits(self, *qubits):
        return self