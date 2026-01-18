from collections.abc import Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit
import itertools
import re
def _mapping_equal(one, two):
    """
    Check if two mappings are equal using the semantics of `equal`.
    """
    if len(one) != len(two):
        return False
    return all((key in two and equal(value, two[key]) for key, value in one.items()))