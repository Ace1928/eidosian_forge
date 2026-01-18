import collections
import copy
import functools
import itertools
import operator
from heat.common import exception
from heat.engine import function
from heat.engine import properties
def _hash_data(data):
    """Return a stable hash value for an arbitrary parsed-JSON data snippet."""
    if isinstance(data, function.Function):
        data = copy.deepcopy(data)
    if not isinstance(data, str):
        if isinstance(data, collections.abc.Sequence):
            return hash(tuple((_hash_data(d) for d in data)))
        if isinstance(data, collections.abc.Mapping):
            item_hashes = (hash(k) ^ _hash_data(v) for k, v in data.items())
            return functools.reduce(operator.xor, item_hashes, 0)
    return hash(data)