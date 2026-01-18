from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pytest
import cirq
class HasKraus(cirq.testing.SingleQubitGate):

    def _has_kraus_(self) -> bool:
        return True