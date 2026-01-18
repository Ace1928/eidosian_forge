from collections import Counter
from contextlib import suppress
from typing import NamedTuple
import numpy as np
from . import is_scalar_nan
def _unique_python(values, *, return_inverse, return_counts):
    try:
        uniques_set = set(values)
        uniques_set, missing_values = _extract_missing(uniques_set)
        uniques = sorted(uniques_set)
        uniques.extend(missing_values.to_list())
        uniques = np.array(uniques, dtype=values.dtype)
    except TypeError:
        types = sorted((t.__qualname__ for t in set((type(v) for v in values))))
        raise TypeError(f'Encoders require their input argument must be uniformly strings or numbers. Got {types}')
    ret = (uniques,)
    if return_inverse:
        ret += (_map_to_integer(values, uniques),)
    if return_counts:
        ret += (_get_counts(values, uniques),)
    return ret[0] if len(ret) == 1 else ret