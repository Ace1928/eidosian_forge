from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
Backward difference coding.

    This coding scheme is useful for ordered factors, and compares the mean of
    each level with the preceding level. So you get the second level minus the
    first, the third level minus the second, etc.

    For full-rank coding, a standard intercept term is added (which gives the
    mean value for the first level).

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Diff)", balanced(a=3))
       # Full rank
       dmatrix("0 + C(a, Diff)", balanced(a=3))
    