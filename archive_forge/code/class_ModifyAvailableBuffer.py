import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class ModifyAvailableBuffer:

    def _apply_unitary_(self, args):
        zz = args.subspace_index(0)
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        oo = args.subspace_index(3)
        args.available_buffer[zz] = args.target_tensor[zz]
        args.available_buffer[zo] = args.target_tensor[oz]
        args.available_buffer[oz] = args.target_tensor[zo]
        args.available_buffer[oo] = args.target_tensor[oo]
        args.available_buffer[...] *= 1j
        args.target_tensor[...] = 99
        return args.available_buffer