import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def _wrap_cop(c: cirq.FrozenCircuit, *tags) -> cirq.FrozenCircuit:
    return cirq.FrozenCircuit(cirq.CircuitOperation(c).with_tags(*tags, t_all))