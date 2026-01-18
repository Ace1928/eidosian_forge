import pytest
import numpy as np
import cirq
class ReturnsNotImplementedUnitary:

    def _unitary_(self):
        return NotImplemented

    def _has_unitary_(self):
        return NotImplemented