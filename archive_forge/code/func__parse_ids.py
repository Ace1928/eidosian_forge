from functools import wraps
from inspect import signature
import warnings
import numpy as np
from autoray import numpy as anp
import pennylane as qml
def _parse_ids(ids, info_dict):
    """Parse different formats of ``ids`` into the right dictionary format,
    potentially using the information in ``info_dict`` to complete it.
    """
    if ids is None:
        return {outer_key: inner_dict.keys() for outer_key, inner_dict in info_dict.items()}
    if isinstance(ids, str):
        return {ids: info_dict[ids].keys()}
    if not isinstance(ids, dict):
        return {_id: info_dict[_id].keys() for _id in ids}
    return ids