from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def DivideAndCeil(dividend, divisor):
    """Returns ceil(dividend / divisor).

  Takes care to avoid the pitfalls of floating point arithmetic that could
  otherwise yield the wrong result for large numbers.

  Args:
    dividend: Dividend for the operation.
    divisor: Divisor for the operation.

  Returns:
    Quotient.
  """
    quotient = dividend // divisor
    if dividend % divisor != 0:
        quotient += 1
    return quotient