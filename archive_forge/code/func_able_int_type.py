from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def able_int_type(values):
    """Find the smallest integer numpy type to contain sequence `values`

    Prefers uint to int if minimum is >= 0

    Parameters
    ----------
    values : sequence
        sequence of integer values

    Returns
    -------
    itype : None or numpy type
        numpy integer type or None if no integer type holds all `values`

    Examples
    --------
    >>> able_int_type([0, 1]) == np.uint8
    True
    >>> able_int_type([-1, 1]) == np.int8
    True
    """
    if any([v % 1 for v in values]):
        return None
    mn = min(values)
    mx = max(values)
    if mn >= 0:
        for ityp in sctypes['uint']:
            if mx <= np.iinfo(ityp).max:
                return ityp
    for ityp in sctypes['int']:
        info = np.iinfo(ityp)
        if mn >= info.min and mx <= info.max:
            return ityp
    return None