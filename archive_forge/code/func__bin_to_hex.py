import numpy as np
from qiskit.exceptions import QiskitError
def _bin_to_hex(bitstring):
    """Convert bitstring readouts (memory) to hexadecimal readouts."""
    return hex(int(bitstring, 2))