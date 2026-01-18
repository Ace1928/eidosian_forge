import numpy as np
from qiskit.exceptions import QiskitError
def _hex_to_bin(hexstring):
    """Convert hexadecimal readouts (memory) to binary readouts."""
    return str(bin(int(hexstring, 16)))[2:]