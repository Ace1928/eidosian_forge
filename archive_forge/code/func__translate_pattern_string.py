import warnings
import numpy as np
import xarray as xr
def _translate_pattern_string(subscripts):
    """Translate a pattern given as string of dimension names to list of dimension names."""
    if '->' in subscripts:
        in_subscripts, out_subscript = subscripts.split('->')
    else:
        in_subscripts = subscripts
        out_subscript = None
    in_dims = [[dim.strip(', ') for dim in in_subscript.split(' ')] for in_subscript in in_subscripts.split(',')]
    if out_subscript is None:
        dims = in_dims
    elif not out_subscript:
        dims = [*in_dims, []]
    else:
        out_dims = [dim.strip(', ') for dim in out_subscript.split(' ')]
        dims = [*in_dims, out_dims]
    return dims