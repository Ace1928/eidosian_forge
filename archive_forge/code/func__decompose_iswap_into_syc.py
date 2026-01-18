from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_iswap_into_syc(a: cirq.Qid, b: cirq.Qid):
    """Decomposes `cirq.ISWAP` into sycamore gates using precomputed coefficients.

    This should only be called when exponent of `cirq.ISwapPowGate` is 1. Other cases are currently
    not supported.

    Args:
        a: First qubit to operate on.
        b: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the `cirq.ISWAP` gate using Sycamore gates.
    """
    yield cirq.PhasedXPowGate(phase_exponent=-0.27250925776964596, exponent=0.2893438375555899).on(a)
    yield ops.SYC(a, b)
    yield (cirq.PhasedXPowGate(phase_exponent=0.8487591858680898, exponent=0.9749387200813147).on(b),)
    yield (cirq.PhasedXPowGate(phase_exponent=-0.3582574564210601).on(a),)
    yield ops.SYC(a, b)
    yield (cirq.PhasedXPowGate(phase_exponent=0.9675022326694225, exponent=0.6908986856555526).on(a),)
    yield (ops.SYC(a, b),)
    yield (cirq.PhasedXPowGate(phase_exponent=0.9161706861686068, exponent=0.14818318325264102).on(b),)
    yield (cirq.PhasedXPowGate(phase_exponent=0.9408341606787907).on(a),)
    yield ((cirq.Z ** (-1.1551880579397293)).on(b),)
    yield ((cirq.Z ** 0.31848343246696365).on(a),)