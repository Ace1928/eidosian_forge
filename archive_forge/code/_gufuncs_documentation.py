import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal

        Apply a generalized ufunc.

        Args:
            args: Input arguments. Each of them can be a :class:`cupy.ndarray`
                object or a scalar. The output arguments can be omitted or be
                specified by the ``out`` argument.
            axes (List of tuples of int, optional):
                A list of tuples with indices of axes a generalized ufunc
                should operate on.
                For instance, for a signature of ``'(i,j),(j,k)->(i,k)'``
                appropriate for matrix multiplication, the base elements are
                two-dimensional matrices and these are taken to be stored in
                the two last axes of each argument.  The corresponding
                axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
                For simplicity, for generalized ufuncs that operate on
                1-dimensional arrays (vectors), a single integer is accepted
                instead of a single-element tuple, and for generalized ufuncs
                for which all outputs are scalars, the output tuples
                can be omitted.
            axis (int, optional):
                A single axis over which a generalized ufunc should operate.
                This is a short-cut for ufuncs that operate over a single,
                shared core dimension, equivalent to passing in axes with
                entries of (axis,) for each single-core-dimension argument
                and ``()`` for all others.
                For instance, for a signature ``'(i),(i)->()'``, it is
                equivalent to passing in ``axes=[(axis,), (axis,), ()]``.
            keepdims (bool, optional):
                If this is set to True, axes which are reduced over will be
                left in the result as a dimension with size one, so that the
                result will broadcast correctly against the inputs. This
                option can only be used for generalized ufuncs that operate
                on inputs that all have the same number of core dimensions
                and with outputs that have no core dimensions , i.e., with
                signatures like ``'(i),(i)->()'`` or ``'(m,m)->()'``.
                If used, the location of the dimensions in the output can
                be controlled with axes and axis.
            casting (str, optional):
                Provides a policy for what kind of casting is permitted.
                Defaults to ``'same_kind'``
            dtype (dtype, optional):
                Overrides the dtype of the calculation and output arrays.
                Similar to signature.
            signature (str or tuple of dtype, optional):
                Either a data-type, a tuple of data-types, or a special
                signature string indicating the input and output types of a
                ufunc. This argument allows you to provide a specific
                signature for the function to be used if registered in the
                ``signatures`` kwarg of the ``__init__`` method.
                If the loop specified does not exist for the ufunc, then
                a TypeError is raised. Normally, a suitable loop is found
                automatically by comparing the input types with what is
                available and searching for a loop with data-types to
                which all inputs can be cast safely. This keyword argument
                lets you bypass that search and choose a particular loop.
            order (str, optional):
                Specifies the memory layout of the output array. Defaults to
                ``'K'``.``'C'`` means the output should be C-contiguous,
                ``'F'`` means F-contiguous, ``'A'`` means F-contiguous
                if the inputs are F-contiguous and not also not C-contiguous,
                C-contiguous otherwise, and ``'K'`` means to match the element
                ordering of the inputs as closely as possible.
            out (cupy.ndarray): Output array. It outputs to new arrays
                default.

        Returns:
            Output array or a tuple of output arrays.
        