import ipaddress
import math
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from django.core.exceptions import ValidationError
from django.utils.deconstruct import deconstructible
from django.utils.encoding import punycode
from django.utils.ipv6 import is_valid_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
@deconstructible
class DecimalValidator:
    """
    Validate that the input does not exceed the maximum number of digits
    expected, otherwise raise ValidationError.
    """
    messages = {'invalid': _('Enter a number.'), 'max_digits': ngettext_lazy('Ensure that there are no more than %(max)s digit in total.', 'Ensure that there are no more than %(max)s digits in total.', 'max'), 'max_decimal_places': ngettext_lazy('Ensure that there are no more than %(max)s decimal place.', 'Ensure that there are no more than %(max)s decimal places.', 'max'), 'max_whole_digits': ngettext_lazy('Ensure that there are no more than %(max)s digit before the decimal point.', 'Ensure that there are no more than %(max)s digits before the decimal point.', 'max')}

    def __init__(self, max_digits, decimal_places):
        self.max_digits = max_digits
        self.decimal_places = decimal_places

    def __call__(self, value):
        digit_tuple, exponent = value.as_tuple()[1:]
        if exponent in {'F', 'n', 'N'}:
            raise ValidationError(self.messages['invalid'], code='invalid', params={'value': value})
        if exponent >= 0:
            digits = len(digit_tuple)
            if digit_tuple != (0,):
                digits += exponent
            decimals = 0
        elif abs(exponent) > len(digit_tuple):
            digits = decimals = abs(exponent)
        else:
            digits = len(digit_tuple)
            decimals = abs(exponent)
        whole_digits = digits - decimals
        if self.max_digits is not None and digits > self.max_digits:
            raise ValidationError(self.messages['max_digits'], code='max_digits', params={'max': self.max_digits, 'value': value})
        if self.decimal_places is not None and decimals > self.decimal_places:
            raise ValidationError(self.messages['max_decimal_places'], code='max_decimal_places', params={'max': self.decimal_places, 'value': value})
        if self.max_digits is not None and self.decimal_places is not None and (whole_digits > self.max_digits - self.decimal_places):
            raise ValidationError(self.messages['max_whole_digits'], code='max_whole_digits', params={'max': self.max_digits - self.decimal_places, 'value': value})

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.max_digits == other.max_digits and (self.decimal_places == other.decimal_places)