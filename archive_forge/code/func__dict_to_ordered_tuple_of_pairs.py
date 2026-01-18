import re
import operator
from fractions import Fraction
import sys
def _dict_to_ordered_tuple_of_pairs(d):
    """
    >>> _dict_to_ordered_tuple_of_pairs(
    ...      { 'key3':'value3', 'key1':'value1', 'key2':'value2' })
    (('key1', 'value1'), ('key2', 'value2'), ('key3', 'value3'))
    """
    l = list(d.items())
    l.sort(key=lambda x: x[0])
    return tuple(l)