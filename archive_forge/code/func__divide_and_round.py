import time as _time
import math as _math
import sys
from operator import index as _index
def _divide_and_round(a, b):
    """divide a by b and round result to the nearest integer

    When the ratio is exactly half-way between two integers,
    the even integer is returned.
    """
    q, r = divmod(a, b)
    r *= 2
    greater_than_half = r > b if b > 0 else r < b
    if greater_than_half or (r == b and q % 2 == 1):
        q += 1
    return q