from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_cz_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.CZ` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.CZPowGate` is 1. Otherwise,
    `_decompose_cphase_into_syc` should be called.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.CZ` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=0.5678998743900456, exponent=0.5863459345743176).on(a)
    yield cirq.PhasedXPowGate(phase_exponent=0.3549946157441739).on(b)
    yield ops.SYC(a, b)
    yield cirq.PhasedXPowGate(phase_exponent=-0.5154334589432878, exponent=0.5228733015013345).on(b)
    yield cirq.PhasedXPowGate(phase_exponent=0.06774925307475355).on(a)
    yield (ops.SYC(a, b),)
    yield (cirq.PhasedXPowGate(phase_exponent=-0.5987667922766213, exponent=0.4136540654256824).on(a),)
    yield ((cirq.Z ** (-0.9255092746611595)).on(b),)
    yield ((cirq.Z ** (-1.333333333333333)).on(a),)