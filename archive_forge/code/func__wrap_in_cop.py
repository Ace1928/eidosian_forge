from typing import List
import numpy as np
import pytest
import cirq
def _wrap_in_cop(ops: cirq.OP_TREE, tag: str):
    return cirq.CircuitOperation(cirq.FrozenCircuit(ops)).with_tags(tag)