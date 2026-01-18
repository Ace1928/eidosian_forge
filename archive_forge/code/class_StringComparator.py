import numpy as np
from ase.ga import get_raw_score
class StringComparator:
    """Compares the calculated hash strings. These strings should be stored
       in atoms.info['key_value_pairs'][key1] and
       atoms.info['key_value_pairs'][key2] ...
       where the keys should be supplied as parameters i.e.
       StringComparator(key1, key2, ...)
    """

    def __init__(self, *keys):
        self.keys = keys

    def looks_like(self, a1, a2):
        for k in self.keys:
            if a1.info['key_value_pairs'][k] == a2.info['key_value_pairs'][k]:
                return True
        return False