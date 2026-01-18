from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def _unicode_pow_10(significand, mantissa):
    if significand in ('1', '1.0'):
        result = u'10'
    else:
        result = significand + u'·10'
    return result + u''.join(map(_unicode_sup.get, str(int(mantissa))))