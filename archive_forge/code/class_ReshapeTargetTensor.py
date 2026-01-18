import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class ReshapeTargetTensor:

    def _apply_unitary_(self, args):
        zz = args.subspace_index(0)
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        oo = args.subspace_index(3)
        args.available_buffer[zz] = args.target_tensor[zz]
        args.available_buffer[zo] = args.target_tensor[zo]
        args.available_buffer[oz] = args.target_tensor[oz]
        args.available_buffer[oo] = args.target_tensor[oo]
        target = args.target_tensor.transpose(*range(1, len(args.target_tensor.shape)), 0).reshape(args.target_tensor.shape)
        target[zz] = args.available_buffer[zz]
        target[zo] = args.available_buffer[oz]
        target[oz] = args.available_buffer[zo]
        target[oo] = args.available_buffer[oo]
        target[...] *= 1j
        args.available_buffer[...] = 99
        return target