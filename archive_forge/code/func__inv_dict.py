from collections.abc import Iterable
import functools
import inspect
import numbers
import numpy as np
import pennylane as qml
def _inv_dict(d):
    """Reverse a dictionary mapping.

    Returns multimap where the keys are the former values,
    and values are sets of the former keys.

    Args:
        d (dict[a->b]): mapping to reverse

    Returns:
        dict[b->set[a]]: reversed mapping
    """
    ret = {}
    for k, v in d.items():
        ret.setdefault(v, set()).add(k)
    return ret