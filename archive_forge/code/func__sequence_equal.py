from collections.abc import Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit
import itertools
import re
def _sequence_equal(one, two):
    """
    Check if two sequences are equal using the semantics of `equal`.
    """
    if len(one) != len(two):
        return False
    return all((equal(i, j) for i, j in zip(one, two)))