import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class USPhoneNumber(FancyValidator):
    """
    Validates, and converts to ###-###-####, optionally with extension
    (as ext.##...).  Only support US phone numbers.  See
    InternationalPhoneNumber for support for that kind of phone number.

    ::

        >>> p = USPhoneNumber()
        >>> p.to_python('333-3333')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a number, with area code, in the form ###-###-####, optionally with "ext.####"
        >>> p.to_python('555-555-5555')
        '555-555-5555'
        >>> p.to_python('1-393-555-3939')
        '1-393-555-3939'
        >>> p.to_python('321.555.4949')
        '321.555.4949'
        >>> p.to_python('3335550000')
        '3335550000'
    """
    _phoneRE = re.compile('^\\s*(?:1-)?(\\d\\d\\d)[\\- \\.]?(\\d\\d\\d)[\\- \\.]?(\\d\\d\\d\\d)(?:\\s*ext\\.?\\s*(\\d+))?\\s*$', re.I)
    messages = dict(phoneFormat=_('Please enter a number, with area code, in the form ###-###-####, optionally with "ext.####"'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self._phoneRE.search(value)
        if not match:
            raise Invalid(self.message('phoneFormat', state), value, state)
        return value

    def _convert_from_python(self, value, state):
        self.assert_string(value, state)
        match = self._phoneRE.search(value)
        if not match:
            raise Invalid(self.message('phoneFormat', state), value, state)
        result = '%s-%s-%s' % (match.group(1), match.group(2), match.group(3))
        if match.group(4):
            result += ' ext.%s' % match.group(4)
        return result