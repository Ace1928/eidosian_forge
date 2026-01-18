import pytest
import numpy as np
import cirq
class ReturnsGreaterThanUnityProbability:

    def _mixture_(self):
        return ((1.2, 'a'), (0.4, 'b'))