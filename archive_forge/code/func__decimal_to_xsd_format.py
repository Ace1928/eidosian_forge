from suds import *
from suds.xsd import *
from suds.sax.date import *
from suds.xsd.sxbase import XBuiltin
import datetime
import decimal
import sys
@staticmethod
def _decimal_to_xsd_format(value):
    """
        Converts a decimal.Decimal value to its XSD decimal type value.

        Result is a string containing the XSD decimal type's lexical value
        representation. The conversion is done without any precision loss.

        Note that Python's native decimal.Decimal string representation will
        not do here as the lexical representation desired here does not allow
        representing decimal values using float-like `<mantissa>E<exponent>'
        format, e.g. 12E+30 or 0.10006E-12.

        """
    value = XDecimal._decimal_canonical(value)
    negative, digits, exponent = value.as_tuple()
    assert digits
    assert digits[0] != 0 or len(digits) == 1
    result = []
    if negative:
        result.append('-')
    if exponent >= 0:
        result.extend((str(x) for x in digits))
        result.extend('0' * exponent)
        return ''.join(result)
    digit_count = len(digits)
    point_offset = digit_count + exponent
    fractional_digit_count = min(digit_count, -exponent)
    while fractional_digit_count and digits[digit_count - 1] == 0:
        digit_count -= 1
        fractional_digit_count -= 1
    if point_offset <= 0:
        result.append('0')
        if digit_count > 0:
            result.append('.')
            result.append('0' * -point_offset)
            result.extend((str(x) for x in digits[:digit_count]))
    else:
        result.extend((str(x) for x in digits[:point_offset]))
        if point_offset < digit_count:
            result.append('.')
            result.extend((str(x) for x in digits[point_offset:digit_count]))
    return ''.join(result)