from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
class GUFuncCallSteps(metaclass=ABCMeta):
    """
    Implements memory management and kernel launch operations for GUFunc calls.

    One instance of this class is instantiated for each call, and the instance
    is specific to the arguments given to the GUFunc call.

    The base class implements the overall logic; subclasses provide
    target-specific implementations of individual functions.
    """
    __slots__ = ['outputs', 'inputs', '_copy_result_to_host']

    @abstractmethod
    def launch_kernel(self, kernel, nelem, args):
        """Implement the kernel launch"""

    @abstractmethod
    def is_device_array(self, obj):
        """
        Return True if `obj` is a device array for this target, False
        otherwise.
        """

    @abstractmethod
    def as_device_array(self, obj):
        """
        Return `obj` as a device array on this target.

        May return `obj` directly if it is already on the target.
        """

    @abstractmethod
    def to_device(self, hostary):
        """
        Copy `hostary` to the device and return the device array.
        """

    @abstractmethod
    def allocate_device_array(self, shape, dtype):
        """
        Allocate a new uninitialized device array with the given shape and
        dtype.
        """

    def __init__(self, nin, nout, args, kwargs):
        outputs = kwargs.get('out')
        if outputs is None and len(args) not in (nin, nin + nout):

            def pos_argn(n):
                return f'{n} positional argument{'s' * (n != 1)}'
            msg = f'This gufunc accepts {pos_argn(nin)} (when providing input only) or {pos_argn(nin + nout)} (when providing input and output). Got {pos_argn(len(args))}.'
            raise TypeError(msg)
        if outputs is not None and len(args) > nin:
            raise ValueError("cannot specify argument 'out' as both positional and keyword")
        else:
            outputs = [outputs] * nout
        all_user_outputs_are_host = True
        self.outputs = []
        for output in outputs:
            if self.is_device_array(output):
                self.outputs.append(self.as_device_array(output))
                all_user_outputs_are_host = False
            else:
                self.outputs.append(output)
        all_host_arrays = not any([self.is_device_array(a) for a in args])
        self._copy_result_to_host = all_host_arrays and all_user_outputs_are_host

        def normalize_arg(a):
            if self.is_device_array(a):
                convert = self.as_device_array
            else:
                convert = np.asarray
            return convert(a)
        normalized_args = [normalize_arg(a) for a in args]
        self.inputs = normalized_args[:nin]
        unused_inputs = normalized_args[nin:]
        if unused_inputs:
            self.outputs = unused_inputs

    def adjust_input_types(self, indtypes):
        """
        Attempt to cast the inputs to the required types if necessary
        and if they are not device arrays.

        Side effect: Only affects the elements of `inputs` that require
        a type cast.
        """
        for i, (ity, val) in enumerate(zip(indtypes, self.inputs)):
            if ity != val.dtype:
                if not hasattr(val, 'astype'):
                    msg = 'compatible signature is possible by casting but {0} does not support .astype()'.format(type(val))
                    raise TypeError(msg)
                self.inputs[i] = val.astype(ity)

    def prepare_outputs(self, schedule, outdtypes):
        """
        Returns a list of output parameters that all reside on the target device.

        Outputs that were passed-in to the GUFunc are used if they reside on the
        device; other outputs are allocated as necessary.
        """
        outputs = []
        for shape, dtype, output in zip(schedule.output_shapes, outdtypes, self.outputs):
            if output is None or self._copy_result_to_host:
                output = self.allocate_device_array(shape, dtype)
            outputs.append(output)
        return outputs

    def prepare_inputs(self):
        """
        Returns a list of input parameters that all reside on the target device.
        """

        def ensure_device(parameter):
            if self.is_device_array(parameter):
                convert = self.as_device_array
            else:
                convert = self.to_device
            return convert(parameter)
        return [ensure_device(p) for p in self.inputs]

    def post_process_outputs(self, outputs):
        """
        Moves the given output(s) to the host if necessary.

        Returns a single value (e.g. an array) if there was one output, or a
        tuple of arrays if there were multiple. Although this feels a little
        jarring, it is consistent with the behavior of GUFuncs in general.
        """
        if self._copy_result_to_host:
            outputs = [self.to_host(output, self_output) for output, self_output in zip(outputs, self.outputs)]
        elif self.outputs[0] is not None:
            outputs = self.outputs
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)