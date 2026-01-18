import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class ReshapeAvailableBuffer:

    def _apply_unitary_(self, args):
        zz = args.subspace_index(0)
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        oo = args.subspace_index(3)
        output = args.available_buffer.transpose(*range(1, len(args.available_buffer.shape)), 0).reshape(args.available_buffer.shape)
        output[zz] = args.target_tensor[zz]
        output[zo] = args.target_tensor[oz]
        output[oz] = args.target_tensor[zo]
        output[oo] = args.target_tensor[oo]
        output[...] *= 1j
        args.target_tensor[...] = 99
        return output