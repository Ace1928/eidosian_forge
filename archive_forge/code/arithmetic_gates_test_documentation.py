import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
Returns a dict of size `2**n` mapping each integer in range [0, 2**n) to itself.