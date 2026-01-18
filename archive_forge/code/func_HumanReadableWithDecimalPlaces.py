from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import re
import six
def HumanReadableWithDecimalPlaces(number, decimal_places=1):
    """Creates a human readable format for bytes with fixed decimal places.

  Args:
    number: The number of bytes.
    decimal_places: The number of decimal places.
  Returns:
    String representing a readable format for number with decimal_places
     decimal places.
  """
    number_format = MakeHumanReadable(number).split()
    num = str(int(round(10 ** decimal_places * float(number_format[0]))))
    if num == '0':
        number_format[0] = '0' + ('.' + '0' * decimal_places if decimal_places else '')
    else:
        num_length = len(num)
        if decimal_places:
            num = num[:num_length - decimal_places] + '.' + num[num_length - decimal_places:]
        number_format[0] = num
    return ' '.join(number_format)