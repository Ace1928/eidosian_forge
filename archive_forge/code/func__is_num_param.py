import re
import sys
from pprint import pprint
def _is_num_param(names, values, to_float=False):
    """
    Return numbers from inputs or raise VdtParamError.
    
    Lets ``None`` pass through.
    Pass in keyword argument ``to_float=True`` to
    use float for the conversion rather than int.
    
    >>> _is_num_param(('', ''), (0, 1.0))
    [0, 1]
    >>> _is_num_param(('', ''), (0, 1.0), to_float=True)
    [0.0, 1.0]
    >>> _is_num_param(('a'), ('a'))
    Traceback (most recent call last):
    VdtParamError: passed an incorrect value "a" for parameter "a".
    """
    fun = to_float and float or int
    out_params = []
    for name, val in zip(names, values):
        if val is None:
            out_params.append(val)
        elif isinstance(val, (int, long, float, string_type)):
            try:
                out_params.append(fun(val))
            except ValueError as e:
                raise VdtParamError(name, val)
        else:
            raise VdtParamError(name, val)
    return out_params