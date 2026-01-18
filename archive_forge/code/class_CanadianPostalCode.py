import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class CanadianPostalCode(Regex):
    """
    Canadian Postal codes.

    ::

        >>> CanadianPostalCode.to_python('V3H 1Z7')
        'V3H 1Z7'
        >>> CanadianPostalCode.to_python('v3h1z7')
        'V3H 1Z7'
        >>> CanadianPostalCode.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (LnL nLn)
    """
    format = _('LnL nLn')
    regex = re.compile('^([a-zA-Z]\\d[a-zA-Z])\\s?(\\d[a-zA-Z]\\d)$')
    strip = True
    messages = dict(invalid=_('Please enter a zip code (%(format)s)'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self.regex.search(value)
        if not match:
            raise Invalid(self.message('invalid', state, format=self.format), value, state)
        return '%s %s' % (match.group(1).upper(), match.group(2).upper())