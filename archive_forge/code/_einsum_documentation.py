import copy
import itertools
import operator
import string
import warnings
import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path
from cupy.linalg._einsum_cutn import _try_use_cutensornet
einsum(subscripts, *operands, dtype=None, optimize=False)

    Evaluates the Einstein summation convention on the operands.
    Using the Einstein summation convention, many common multi-dimensional
    array operations can be represented in a simple fashion. This function
    provides a way to compute such summations.

    .. note::

       - Memory contiguity of the returned array is not always compatible with
         that of :func:`numpy.einsum`.
       - ``out``, ``order``, and ``casting`` options are not supported.
       - If :envvar:`CUPY_ACCELERATORS` includes ``cutensornet``, the `einsum`
         calculation will be performed by the cuTensorNet backend if possible.

           - The support of the ``optimize`` option is limited (currently, only
             `False`, 'cutensornet', or a custom path for pairwise contraction
             is supported, and the maximum intermediate size is ignored). If
             you need finer control for path optimization, consider replacing
             :func:`cupy.einsum` by :func:`cuquantum.contract` instead.
           - Requires `cuQuantum Python`_ (v22.03+).

       - If :envvar:`CUPY_ACCELERATORS` includes ``cutensor``, `einsum` will be
         accelerated by the cuTENSOR backend whenever possible.

    Args:
        subscripts (str): Specifies the subscripts for summation.
        operands (sequence of arrays): These are the arrays for the operation.
        dtype: If provided, forces the calculation to use the data type
            specified. Default is None.
        optimize: Valid options include {`False`, `True`, 'greedy', 'optimal'}.
            Controls if intermediate optimization should occur. No optimization
            will occur if `False`, and `True` will default to the 'greedy'
            algorithm. Also accepts an explicit contraction list from
            :func:`numpy.einsum_path`. Defaults to `False`. If a pair is
            supplied, the second argument is assumed to be the maximum
            intermediate size created.

    Returns:
        cupy.ndarray:
            The calculation based on the Einstein summation convention.

    .. seealso:: :func:`numpy.einsum`
    .. _cuQuantum Python: https://docs.nvidia.com/cuda/cuquantum/python/
    