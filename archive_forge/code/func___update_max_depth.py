import collections
import dns.name
from ._compat import xrange
def __update_max_depth(self, key):
    if len(key) == self.max_depth:
        self.max_depth_items = self.max_depth_items + 1
    elif len(key) > self.max_depth:
        self.max_depth = len(key)
        self.max_depth_items = 1