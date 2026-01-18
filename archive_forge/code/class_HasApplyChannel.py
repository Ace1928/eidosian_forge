import numpy as np
import pytest
import cirq
class HasApplyChannel:

    def _apply_channel_(self, args: cirq.ApplyChannelArgs):
        args.target_tensor = 0.5 * args.target_tensor + 0.5 * np.dot(np.dot(x, args.target_tensor), x)
        return args.target_tensor