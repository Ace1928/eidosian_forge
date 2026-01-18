from typing import List
import pytest
import sympy
import numpy as np
import cirq
import cirq_google as cg
def _too_many_reps(circuits: List[cirq.Circuit], sweeps: List[cirq.Sweepable], repetitions: int):
    if repetitions > 10000:
        raise ValueError('Too many repetitions')