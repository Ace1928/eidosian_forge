import numpy as np
import pytest
import cirq
class ReturnsExpansion:

    def __init__(self, expansion: cirq.LinearDict[str]) -> None:
        self._expansion = expansion

    def _pauli_expansion_(self) -> cirq.LinearDict[str]:
        return self._expansion