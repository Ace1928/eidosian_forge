import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class CreateNewBuffer:

    def _apply_unitary_(self, args):
        u = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=args.target_tensor.dtype) * 1j
        new_shape = args.target_tensor.shape[:-2] + (4, 1)
        ret = np.matmul(u, args.target_tensor.reshape(new_shape)).reshape(args.target_tensor.shape)
        args.target_tensor[...] = 99
        args.available_buffer[...] = 98
        return ret