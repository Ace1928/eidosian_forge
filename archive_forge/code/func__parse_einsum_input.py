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
def _parse_einsum_input(args):
    """Parse einsum operands.

    This function is based on `numpy.core.einsumfunc._parse_einsum_input`
    function in NumPy 1.14.

    Parameters
    ----------
    args : tuple
        The non-keyword arguments to einsum

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    (['@a, @a'], 'xz', [a, b])

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    (['@a, @a'], 'xz', [a, b])
    """
    if len(args) == 0:
        raise ValueError('must specify the einstein sum subscripts string and at least one operand, or at least one operand and its corresponding subscripts list')
    if isinstance(args[0], str):
        subscripts = args[0]
        operands = list(args[1:])
        for s in subscripts:
            if s in '.,-> ':
                continue
            if s not in einsum_symbols:
                raise ValueError("invalid subscript '%s' in einstein sum subscripts string, subscripts must be letters" % s)
        subscripts = subscripts.replace('...', '@')
        if '.' in subscripts:
            raise ValueError("einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')")
        if '-' in subscripts or '>' in subscripts:
            invalid = subscripts.count('-') > 1 or subscripts.count('>') > 1
            subscripts = subscripts.split('->')
            if invalid or len(subscripts) != 2:
                raise ValueError("einstein sum subscript string does not contain proper '->' output specified")
            input_subscripts, output_subscript = subscripts
            output_subscript = output_subscript.replace(' ', '')
        else:
            input_subscripts = subscripts
            output_subscript = None
        input_subscripts = input_subscripts.replace(' ', '').split(',')
        if len(input_subscripts) != len(operands):
            msg = 'more' if len(operands) > len(input_subscripts) else 'fewer'
            raise ValueError(msg + ' operands provided to einstein sum function than specified in the subscripts string')
    else:
        args = list(args)
        operands = []
        input_subscripts = []
        while len(args) >= 2:
            operands.append(args.pop(0))
            input_subscripts.append(_parse_int_subscript(args.pop(0)))
        if args:
            output_subscript = _parse_int_subscript(args[0])
        else:
            output_subscript = None
    return (input_subscripts, output_subscript, operands)