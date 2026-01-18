import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _label_suffix(label, suffix):
    """Returns (label + suffix) or a truncated version if it's too long.
    Parameters
    ----------
    label : str
        Label name
    suffix : str
        Label suffix
    """
    if len(label) > 50:
        nhead = 25
        return ''.join([label[:nhead], '..', suffix])
    else:
        return label + suffix