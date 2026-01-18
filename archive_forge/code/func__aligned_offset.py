import numpy as np
def _aligned_offset(offset, alignment):
    return -(-offset // alignment) * alignment