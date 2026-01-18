import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _decimal_lshift_exact(n, e):
    """ Given integers n and e, return n * 10**e if it's an integer, else None.

    The computation is designed to avoid computing large powers of 10
    unnecessarily.

    >>> _decimal_lshift_exact(3, 4)
    30000
    >>> _decimal_lshift_exact(300, -999999999)  # returns None

    """
    if n == 0:
        return 0
    elif e >= 0:
        return n * 10 ** e
    else:
        str_n = str(abs(n))
        val_n = len(str_n) - len(str_n.rstrip('0'))
        return None if val_n < -e else n // 10 ** (-e)