import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
Multi-qubit operation with unitary.

        Used to verify that `is_supported_operation` does not attempt to
        allocate the unitary for multi-qubit operations.
        