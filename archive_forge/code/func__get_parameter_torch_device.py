import warnings
import inspect
import logging
import semantic_version
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from . import DefaultQubitLegacy
@staticmethod
def _get_parameter_torch_device(ops):
    """An auxiliary function to determine the Torch device specified for
        the gate parameters of the input operations.

        Returns the first CUDA Torch device found (if any) using a string
        format. Does not handle tensors put on multiple CUDA Torch devices.
        Such a case raises an error with Torch.

        If CUDA is not used with any of the parameters, then specifies the CPU
        if the parameters are on the CPU or None if there were no parametric
        operations.

        Args:
            ops (list[Operator]): list of operations to check

        Returns:
            str or None: The string of the Torch device determined or None if
            there is no data for any operations.
        """
    par_torch_device = None
    for op in ops:
        for data in op.data:
            if hasattr(data, 'is_cuda'):
                if data.is_cuda:
                    return ':'.join([data.device.type, str(data.device.index)])
                par_torch_device = 'cpu'
    return par_torch_device