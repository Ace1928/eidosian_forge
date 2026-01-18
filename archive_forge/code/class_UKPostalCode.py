import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class UKPostalCode(Regex):
    """
    UK Postal codes. Please see BS 7666.

    ::

        >>> UKPostalCode.to_python('BFPO 3')
        'BFPO 3'
        >>> UKPostalCode.to_python('LE11 3GR')
        'LE11 3GR'
        >>> UKPostalCode.to_python('l1a 3gr')
        'L1A 3GR'
        >>> UKPostalCode.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a valid postal code (for format see BS 7666)
    """
    regex = re.compile('^((ASCN|BBND|BIQQ|FIQQ|PCRN|SIQQ|STHL|TDCU|TKCA)\\s?1ZZ|BFPO (c\\/o )?[1-9]{1,4}|GIR\\s?0AA|[A-PR-UWYZ]([0-9]{1,2}|([A-HK-Y][0-9]|[A-HK-Y][0-9]([0-9]|[ABEHMNPRV-Y]))|[0-9][A-HJKS-UW])\\s?[0-9][ABD-HJLNP-UW-Z]{2})$', re.I)
    strip = True
    messages = dict(invalid=_('Please enter a valid postal code (for format see BS 7666)'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self.regex.search(value)
        if not match:
            raise Invalid(self.message('invalid', state), value, state)
        value = match.group(1).upper()
        if not value.startswith('BFPO'):
            value = ''.join(value.split())
            value = '%s %s' % (value[:-3], value[-3:])
        return value