import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
class _GUFunc:
    """
    Creates a Generalized Universal Function by wrapping a user
    provided function with the signature.

    ``signature`` determines if the function consumes or produces core
    dimensions. The remaining dimensions in given input arrays (``*args``)
    are considered loop dimensions and are required to broadcast
    naturally against each other.

    Args:
        func (callable):
            Function to call like ``func(*args, **kwargs)`` on input arrays
            (``*args``) that returns an array or tuple of arrays. If
            multiple arguments with non-matching dimensions are supplied,
            this function is expected to vectorize (broadcast) over axes of
            positional arguments in the style of NumPy universal functions.
        signature (string):
            Specifies what core dimensions are consumed and produced by
            ``func``.  According to the specification of numpy.gufunc
            signature.
        supports_batched (bool, optional):
            If the wrapped function supports to pass the complete input
            array with the loop and the core dimensions.
            Defaults to `False`. Dimensions will be iterated in the
            `GUFunc` processing code.
        supports_out (bool, optional):
            If the wrapped function supports out as one of its kwargs.
            Defaults to `False`.
        signatures (list of tuple of str):
            Contains strings in the form of 'ii->i' with i being the char of a
            dtype. Each element of the list is a tuple with the string
            and a alternative function to `func` to be executed when the inputs
            of the function can be casted as described by this function.
        name (str, optional):
            Name for the GUFunc object. If not specified, ``func``'s name
            is used.
        doc (str, optional):
            Docstring for the GUFunc object. If not specified, ``func.__doc__``
            is used.
    """

    def __init__(self, func, signature, **kwargs):
        self._func = func
        self._signature = signature
        self.__name__ = kwargs.pop('name', func.__name__)
        self.__doc__ = kwargs.pop('doc', func.__doc__)
        self._supports_batched = kwargs.pop('supports_batched', False)
        self._supports_out = kwargs.pop('supports_out', False)
        signatures = kwargs.pop('signatures', [])
        if kwargs:
            raise TypeError('got unexpected keyword arguments: ' + ', '.join([repr(k) for k in kwargs]))
        input_coredimss, output_coredimss = _parse_gufunc_signature(self._signature)
        self._input_coredimss = input_coredimss
        self._output_coredimss = output_coredimss
        self._min_dims = [0] * len(input_coredimss)
        for i, inp in enumerate(input_coredimss):
            for d in inp:
                if d[-1] != '?':
                    self._min_dims[i] += 1
        self._nout = 0 if not isinstance(output_coredimss, list) else len(output_coredimss)
        self._nin = 0 if not isinstance(input_coredimss, list) else len(input_coredimss)
        self._ops_register = _OpsRegister(signatures, self._func, self._nin, self._nout, self.__name__)

    def _apply_func_to_inputs(self, func, dim, sizes, dims, args, outs):
        if self._supports_batched or dim == len(dims):
            if self._supports_out and outs is not None:
                outs = outs[0] if len(outs) == 1 else outs
                func(*args, out=outs)
            else:
                fouts = func(*args)
                if isinstance(fouts, cupy.ndarray):
                    fouts = (fouts,)
                for o, fo in zip(outs, fouts):
                    cupy._core.elementwise_copy(fo, o)
        else:
            dim_size = sizes[dims[dim]][0]
            for i in range(dim_size):
                n_args = [a[i] for a in args]
                if outs is not None:
                    n_outs = [o[i] for o in outs]
                    self._apply_func_to_inputs(func, dim + 1, sizes, dims, n_args, n_outs)

    def _transpose_element(self, arg, iax, shape):
        iax = tuple((a if a < 0 else a - len(shape) for a in iax))
        tidc = tuple((i for i in range(-len(shape) + 0, 0) if i not in iax)) + iax
        return arg.transpose(tidc)

    def _get_args_transposed(self, args, input_axes, outs, output_axes):
        transposed_args = []
        missing_dims = set()
        for i, (arg, iax, input_coredims, md) in enumerate(zip(args, input_axes, self._input_coredimss, self._min_dims)):
            shape = arg.shape
            nds = len(shape)
            if nds < md:
                raise ValueError(f'Input operand {i} does not have enough dimensions (has {nds}, gufunc core with signature {self._signature} requires {md}')
            optionals = len(input_coredims) - nds
            if optionals > 0:
                if input_coredims[0][-1] == '?':
                    shape = (1,) * optionals + shape
                    missing_dims.update(set(input_coredims[:optionals]))
                else:
                    shape = shape + (1,) * optionals
                    missing_dims.update(set(input_coredims[min(0, len(shape) - 1):]))
                arg = arg.reshape(shape)
            transposed_args.append(self._transpose_element(arg, iax, shape))
        args = transposed_args
        if outs is not None:
            transposed_outs = []
            for out, iox, coredims in zip(outs, output_axes, self._output_coredimss):
                transposed_outs.append(self._transpose_element(out, iox, out.shape))
            if len(transposed_outs) == len(outs):
                outs = transposed_outs
        shape = internal._broadcast_shapes([a.shape[:-len(self._input_coredimss)] for a in args])
        args = [_manipulation.broadcast_to(a, shape + a.shape[-len(self._input_coredimss):]) for a in args]
        input_shapes = [a.shape for a in args]
        num_loopdims = [len(s) - len(cd) for s, cd in zip(input_shapes, self._input_coredimss)]
        max_loopdims = max(num_loopdims) if num_loopdims else None
        core_input_shapes = [dict(zip(icd, s[n:])) for s, n, icd in zip(input_shapes, num_loopdims, self._input_coredimss)]
        core_shapes = {}
        for d in core_input_shapes:
            core_shapes.update(d)
        loop_input_dimss = [tuple(('__loopdim%d__' % d for d in range(max_loopdims - n, max_loopdims))) for n in num_loopdims]
        input_dimss = [li + c for li, c in zip(loop_input_dimss, self._input_coredimss)]
        loop_output_dims = max(loop_input_dimss, key=len, default=())
        dimsizess = {}
        for dims, shape in zip(input_dimss, input_shapes):
            for dim, size in zip(dims, shape):
                dimsizes = dimsizess.get(dim, [])
                dimsizes.append(size)
                dimsizess[dim] = dimsizes
        for dim, sizes in dimsizess.items():
            if set(sizes).union({1}) != {1, max(sizes)}:
                raise ValueError(f'Dimension {dim} with different lengths in arrays')
        return (args, dimsizess, loop_output_dims, outs, missing_dims)

    def _determine_order(self, args, order):
        if order.upper() in ('C', 'K'):
            return 'C'
        elif order.upper() == 'A':
            order = 'F' if all([a.flags.f_contiguous and (not a.flags.c_contiguous) for a in args]) else 'C'
            return order
        elif order.upper() == 'F':
            return 'F'
        else:
            raise RuntimeError(f'Unknown order {order}')

    def __call__(self, *args, **kwargs):
        """
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
        """
        outs = kwargs.pop('out', None)
        axes = kwargs.pop('axes', None)
        axis = kwargs.pop('axis', None)
        order = kwargs.pop('order', 'K')
        dtype = kwargs.pop('dtype', None)
        keepdims = kwargs.pop('keepdims', False)
        signature = kwargs.pop('signature', None)
        casting = kwargs.pop('casting', 'same_kind')
        if len(kwargs) > 0:
            raise RuntimeError('Unknown kwargs {}'.format(' '.join(kwargs.keys())))
        ret_dtype = None
        func = self._func
        args, ret_dtype, func = self._ops_register.determine_dtype(args, dtype, casting, signature)
        if not type(self._signature) == str:
            raise TypeError('`signature` has to be of type string')
        if outs is not None and type(outs) != tuple:
            if isinstance(outs, cupy.ndarray):
                outs = (outs,)
            else:
                raise TypeError('`outs` must be a tuple or `cupy.ndarray`')
        filter_order = self._determine_order(args, order)
        input_coredimss = self._input_coredimss
        output_coredimss = self._output_coredimss
        if outs is not None and type(outs) != tuple:
            raise TypeError('`outs` must be a tuple')
        input_axes, output_axes = _validate_normalize_axes(axes, axis, keepdims, input_coredimss, output_coredimss)
        if len(input_coredimss) != len(args):
            ValueError('According to `signature`, `func` requires %d arguments, but %s given' % (len(input_coredimss), len(args)))
        args, dimsizess, loop_output_dims, outs, m_dims = self._get_args_transposed(args, input_axes, outs, output_axes)
        out_shape = [dimsizess[od][0] for od in loop_output_dims]
        if self._nout > 0:
            out_shape += [dimsizess[od][0] for od in output_coredimss[0]]
        out_shape = tuple(out_shape)
        if outs is None:
            outs = cupy.empty(out_shape, dtype=ret_dtype, order=filter_order)
            if order == 'K':
                strides = internal._get_strides_for_order_K(outs, ret_dtype, out_shape)
                outs._set_shape_and_strides(out_shape, strides, True, True)
            outs = (outs,)
        else:
            if outs[0].shape != out_shape:
                raise ValueError(f'Invalid shape for out {outs[0].shape} needs {out_shape}')
            _raise_if_invalid_cast(ret_dtype, outs[0].dtype, casting, 'out dtype')
        self._apply_func_to_inputs(func, 0, dimsizess, loop_output_dims, args, outs)
        if self._nout == 0:
            output_coredimss = [output_coredimss]
        leaf_arrs = []
        for tmp in outs:
            for i, (ocd, oax) in enumerate(zip(output_coredimss, output_axes)):
                leaf_arr = tmp
                if keepdims:
                    slices = len(leaf_arr.shape) * (slice(None),) + len(oax) * (numpy.newaxis,)
                    leaf_arr = leaf_arr[slices]
                tidcs = [None] * len(leaf_arr.shape)
                for i, oa in zip(range(-len(oax), 0), oax):
                    tidcs[oa] = i
                j = 0
                for i in range(len(tidcs)):
                    if tidcs[i] is None:
                        tidcs[i] = j
                        j += 1
                leaf_arr = leaf_arr.transpose(tidcs)
                if len(m_dims) > 0:
                    shape = leaf_arr.shape
                    core_shape = shape[-len(ocd):]
                    core_shape = tuple([d for d, n in zip(core_shape, ocd) if n not in m_dims])
                    shape = shape[:-len(ocd)] + core_shape
                    leaf_arr = leaf_arr.reshape(shape)
                leaf_arrs.append(leaf_arr)
        return tuple(leaf_arrs) if self._nout > 1 else leaf_arrs[0]