from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
def _cases_for_random_circuit():
    i = 0
    while i < 10:
        n_qubits = random.randint(1, 20)
        n_moments = random.randint(1, 10)
        op_density = random.random()
        if random.randint(0, 1):
            gate_domain = dict(random.sample(tuple(cirq.testing.DEFAULT_GATE_DOMAIN.items()), random.randint(1, len(cirq.testing.DEFAULT_GATE_DOMAIN))))
            if all((n > n_qubits for n in gate_domain.values())):
                continue
        else:
            gate_domain = None
        pass_qubits = random.choice((True, False))
        yield (n_qubits, n_moments, op_density, gate_domain, pass_qubits)
        i += 1