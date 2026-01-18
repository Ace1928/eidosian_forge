import pytest
import numpy as np
import cirq
class ReturnsNonnormalizedTuple:

    def _mixture_(self):
        return ((0.4, 'a'), (0.4, 'b'))