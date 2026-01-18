import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def _assert_pass_over(ops: List[cirq.Operation], before: cirq.PauliString, after: cirq.PauliString):
    assert before.pass_operations_over(ops[::-1]) == after
    assert after.pass_operations_over(ops, after_to_before=True) == before