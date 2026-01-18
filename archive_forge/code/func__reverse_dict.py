from .enums import (
from .constants import (
from ..common.utils import bytes2hex
def _reverse_dict(d, low_priority=()):
    """
    This is a tiny helper function to "reverse" the keys/values of a dictionary
    provided in the first argument, i.e. {k: v} becomes {v: k}.

    The second argument (optional) provides primitive control over what to do in
    the case of conflicting values - if a value is present in this list, it will
    not override any other entries of the same value.
    """
    out = {}
    for k, v in d.items():
        if v in out and k in low_priority:
            continue
        out[v] = k
    return out