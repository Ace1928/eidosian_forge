import sys
import unittest
from libcloud.common.types import LazyList
def _get_more_not_exhausted(self, last_key, value_dict):
    self._get_more_counter += 1
    if not last_key:
        data, last_key, exhausted = ([1, 2, 3, 4, 5], 5, False)
    else:
        data, last_key, exhausted = ([6, 7, 8, 9, 10], 10, True)
    return (data, last_key, exhausted)