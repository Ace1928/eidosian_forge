import numpy as np
from qiskit.exceptions import QiskitError
def _pad_zeros(bitstring, memory_slots):
    """If the bitstring is truncated, pad extra zeros to make its
    length equal to memory_slots"""
    return format(int(bitstring, 2), f'0{memory_slots}b')