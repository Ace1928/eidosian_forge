import pytest
import numpy as np
import cirq
class Bad2:

    def _decompose_(self):
        return [cirq.X(cirq.LineQubit(0))]