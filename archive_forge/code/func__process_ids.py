from collections import OrderedDict
from functools import wraps
from inspect import signature
from itertools import product
import numpy as np
import pennylane as qml
from .utils import get_spectrum, join_spectra
def _process_ids(encoding_args, argnum, qnode):
    """Process the passed ``encoding_args`` and ``argnum`` or infer them from
    the QNode signature.

    Args:
        encoding_args (dict[str, list[tuple]] or set): Parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``. If a ``set``, all values are set to ``Ellipsis``
        argnum (list[int]): Numerical indices for arguments
        qnode (QNode): QNode to infer the ``encoding_args`` and ``argnum`` from
            if both are ``None``
    Returns:
        OrderedDict[str, list[tuple]]: Ordered parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``
        list[int]: Numerical indices for arguments

    In ``qnode_spectrum`` both ``encoding_args`` and ``argnum`` are required.
    However, they can be inferred from one another and even from the QNode signature,
    which is done in this helper function, using the following rules/design choices:

      - If ``argnum`` is provided, the QNode arguments with the indices in ``argnum``
        are considered and added to ``encoding_args`` with an ``Ellipsis``, meaning
        that for array-valued arguments all parameters are considered in
        ``qnode_spectrum``.
      - If ``encoding_args`` is provided and is a dictionary, it is preserved
        up to arguments that do not appear in the QNode. Also, it is converted to
        an ``OrderedDict``, inferring the ordering from the QNode arguments.
        Passing a set with ``keys`` instead is an alias for
        ``{key: ... for key in keys}``.
        ``argnum`` will contain the indices of these arguments.
      - If both ``encoding_args`` and ``argnum`` are passed, ``encoding_args`` takes
        precedence over ``argnum``, in particular ``argnum`` is overwritten.
      - If neither is passed, all arguments of the passed QNode that do not have a
        default value defined are considered
        and their value is an ``Ellipsis``, so that all parameters of array-valued
        arguments will be considered in ``qnode_spectrum``.

    **Example**

    As an example, consider the qnode

    >>> @qml.qnode(dev)
    >>> def circuit(a, b, c, x=2):
    ...     return qml.expval(qml.X(0))

    which takes arguments:

    >>> a = np.array([2.4, 1.2, 3.1])
    >>> b = 0.2
    >>> c = np.arange(20, dtype=float).reshape((2, 5, 2))

    Then we may use the following inputs

    >>> encoding_args = {"a": [(1,), (2,)], "c": ..., "x": [()]}
    >>> argnum = [2, 0]

    in various combinations:

    >>> _process_ids(encoding_args, None, circuit)
    (OrderedDict([('a', [(1,), (2,)]), ('c', Ellipsis), ('x', [()])]), [0, 2, 3])

    The first output, ``encoding_args``, essentially is unchanged, it simply was ordered in
    the order of the QNode arguments. The second output, ``argnum``, contains all three
    argument indices because all of ``a``, ``b``, and ``c`` appear in ``encoding_args``.
    If we in addition pass ``argnum``, it is ignored:

    >>> _process_ids(encoding_args, argnum, circuit)
    (OrderedDict([('a', [(1,), (2,)]), ('c', Ellipsis), ('x', [()])]), [0, 2, 3])

    Only if we leave out ``encoding_args`` does it make a difference:

    >>> _process_ids(None, argnum, circuit)
    (OrderedDict([('a', Ellipsis), ('c', Ellipsis)]), [0, 2])

    Now only the arguments in ``argnum`` are considered, in particular the ``argnum`` input
    is simply sorted. In ``encoding_args``, all argument names are paired with an ``Ellipsis``.
    If we skip both inputs, all QNode arguments are extracted:

    >>> _process_ids(None, None, circuit)
    (OrderedDict([('a', Ellipsis), ('b', Ellipsis), ('c', Ellipsis)]), [0, 1, 2])

    Note that ``x`` does not appear here, because it has a default value defined and thus is
    considered a keyword argument.

    """
    sig_pars = signature(qnode.func).parameters
    arg_names = list(sig_pars.keys())
    arg_names_no_def = [name for name, par in sig_pars.items() if par.default is par.empty]
    if encoding_args is None:
        if argnum is None:
            encoding_args = OrderedDict(((name, ...) for name in arg_names_no_def))
            argnum = list(range(len(arg_names_no_def)))
        elif np.isscalar(argnum):
            encoding_args = OrderedDict({arg_names[argnum]: ...})
            argnum = [argnum]
        else:
            argnum = sorted(argnum)
            encoding_args = OrderedDict(((arg_names[num], ...) for num in argnum))
    else:
        requested_names = set(encoding_args)
        if not all((name in arg_names for name in requested_names)):
            raise ValueError(f'Not all names in {requested_names} are known. Known arguments: {arg_names}')
        if isinstance(encoding_args, set):
            encoding_args = OrderedDict(((name, ...) for name in arg_names if name in requested_names))
        else:
            encoding_args = OrderedDict(((name, encoding_args[name]) for name in arg_names if name in requested_names))
        argnum = [arg_names.index(name) for name in encoding_args]
    return (encoding_args, argnum)