import collections
from typing import Dict, Counter, List, Optional, Sequence
import numpy as np
import cirq
def _pretty_str_dict(value: dict, bit_count: int) -> str:
    """Pretty prints a dict, converting int dict values to bit strings."""
    strs = []
    for k, v in value.items():
        bits = ''.join((str(b) for b in cirq.big_endian_int_to_bits(k, bit_count=bit_count)))
        strs.append(f'{bits}: {v}')
    return '\n'.join(strs)