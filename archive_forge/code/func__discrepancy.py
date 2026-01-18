from collections import defaultdict
from itertools import chain
def _discrepancy(self, other, cb):
    default = self.default_factory()
    _self = self.copy()
    _other = other.copy()
    try:
        for k in set(chain(_self.keys(), _other.keys())):
            if not cb(_self[k], _other.get(k, default)):
                return False
        return True
    except TypeError:
        return False