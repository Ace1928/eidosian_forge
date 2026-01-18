from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def _html_pow_10(significand, mantissa):
    if significand in ('1', '1.0'):
        result = '10<sup>'
    else:
        result = significand + '&sdot;10<sup>'
    return result + str(int(mantissa)) + '</sup>'