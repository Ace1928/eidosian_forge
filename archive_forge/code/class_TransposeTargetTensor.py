import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class TransposeTargetTensor:

    def _apply_unitary_(self, args):
        indices = list(range(len(args.target_tensor.shape)))
        indices[args.axes[0]], indices[args.axes[1]] = (indices[args.axes[1]], indices[args.axes[0]])
        target = args.target_tensor.transpose(*indices)
        target[...] *= 1j
        args.available_buffer[...] = 99
        return target