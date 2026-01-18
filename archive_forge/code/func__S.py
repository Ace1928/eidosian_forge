import numpy as np
import pytest
import cirq
def _S(table, q):
    table.rs[:] ^= table.xs[:, q] & table.zs[:, q]
    table.zs[:, q] ^= table.xs[:, q]