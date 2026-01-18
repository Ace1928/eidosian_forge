from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def _number_to_X(number, uncertainty, unit, fmt, unit_fmt, fmt_pow_10, space=' '):
    uncertainty = uncertainty or getattr(number, 'uncertainty', None)
    unit = unit or unit_of(number)
    integer_one = 1
    if unit is integer_one:
        unit_str = ''
        mag = number
    else:
        unit_str = space + unit_fmt(unit)
        mag = to_unitless(number, unit)
        if uncertainty is not None:
            uncertainty = to_unitless(uncertainty, unit)
    if uncertainty is None:
        if fmt is None:
            fmt = 5
        if isinstance(fmt, int):
            flt = '%%.%dg' % fmt % mag
        else:
            flt = fmt(mag)
    else:
        if fmt is None:
            fmt = 2
        if isinstance(fmt, int):
            flt = _float_str_w_uncert(mag, uncertainty, fmt)
        else:
            flt = fmt(mag, uncertainty)
    if 'e' in flt:
        significand, mantissa = flt.split('e')
        return fmt_pow_10(significand, mantissa) + unit_str
    else:
        return flt + unit_str