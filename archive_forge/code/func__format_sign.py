import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _format_sign(is_negative, spec):
    """Determine sign character."""
    if is_negative:
        return '-'
    elif spec['sign'] in ' +':
        return spec['sign']
    else:
        return ''