import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
class XRI(FancyValidator):
    """
    Validator for XRIs.

    It supports both i-names and i-numbers, of the first version of the XRI
    standard.

    ::

        >>> inames = XRI(xri_type="i-name")
        >>> inames.to_python("   =John.Smith ")
        '=John.Smith'
        >>> inames.to_python("@Free.Software.Foundation")
        '@Free.Software.Foundation'
        >>> inames.to_python("Python.Software.Foundation")
        Traceback (most recent call last):
            ...
        Invalid: The type of i-name is not defined; it may be either individual or organizational
        >>> inames.to_python("http://example.org")
        Traceback (most recent call last):
            ...
        Invalid: The type of i-name is not defined; it may be either individual or organizational
        >>> inames.to_python("=!2C43.1A9F.B6F6.E8E6")
        Traceback (most recent call last):
            ...
        Invalid: "!2C43.1A9F.B6F6.E8E6" is an invalid i-name
        >>> iname_with_schema = XRI(True, xri_type="i-name")
        >>> iname_with_schema.to_python("=Richard.Stallman")
        'xri://=Richard.Stallman'
        >>> inames.to_python("=John Smith")
        Traceback (most recent call last):
            ...
        Invalid: "John Smith" is an invalid i-name
        >>> inumbers = XRI(xri_type="i-number")
        >>> inumbers.to_python("!!1000!de21.4536.2cb2.8074")
        '!!1000!de21.4536.2cb2.8074'
        >>> inumbers.to_python("@!1000.9554.fabd.129c!2847.df3c")
        '@!1000.9554.fabd.129c!2847.df3c'

    """
    iname_valid_pattern = re.compile('\n    ^\n    [\\w]+                  # A global alphanumeric i-name\n    (\\.[\\w]+)*             # An i-name with dots\n    (\\*[\\w]+(\\.[\\w]+)*)*   # A community i-name\n    $\n    ', re.VERBOSE | re.UNICODE)
    iname_invalid_start = re.compile('^[\\d\\.-]', re.UNICODE)
    '@cvar: These characters must not be at the beggining of the i-name'
    inumber_pattern = re.compile("\n    ^\n    (\n    [=@]!       # It's a personal or organization i-number\n    |\n    !!          # It's a network i-number\n    )\n    [\\dA-F]{1,4}(\\.[\\dA-F]{1,4}){0,3}       # A global i-number\n    (![\\dA-F]{1,4}(\\.[\\dA-F]{1,4}){0,3})*   # Zero or more sub i-numbers\n    $\n    ", re.VERBOSE | re.IGNORECASE)
    messages = dict(noType=_('The type of i-name is not defined; it may be either individual or organizational'), repeatedChar=_('Dots and dashes may not be repeated consecutively'), badIname=_('"%(iname)s" is an invalid i-name'), badInameStart=_('i-names may not start with numbers nor punctuation marks'), badInumber=_('"%(inumber)s" is an invalid i-number'), badType=_('The XRI must be a string (not a %(type)s: %(value)r)'), badXri=_('"%(xri_type)s" is not a valid type of XRI'))

    def __init__(self, add_xri=False, xri_type='i-name', **kwargs):
        """Create an XRI validator.

        @param add_xri: Should the schema be added if not present?
            Officially it's optional.
        @type add_xri: C{bool}
        @param xri_type: What type of XRI should be validated?
            Possible values: C{i-name} or C{i-number}.
        @type xri_type: C{str}

        """
        self.add_xri = add_xri
        assert xri_type in ('i-name', 'i-number'), 'xri_type must be "i-name" or "i-number"'
        self.xri_type = xri_type
        super(XRI, self).__init__(**kwargs)

    def _convert_to_python(self, value, state):
        """Prepend the 'xri://' schema if needed and remove trailing spaces"""
        value = value.strip()
        if self.add_xri and (not value.startswith('xri://')):
            value = 'xri://' + value
        return value

    def _validate_python(self, value, state=None):
        """Validate an XRI

        @raise Invalid: If at least one of the following conditions in met:
            - C{value} is not a string.
            - The XRI is not a personal, organizational or network one.
            - The relevant validator (i-name or i-number) considers the XRI
                is not valid.

        """
        if not isinstance(value, str):
            raise Invalid(self.message('badType', state, type=str(type(value)), value=value), value, state)
        if value.startswith('xri://'):
            value = value[6:]
        if not value[0] in ('@', '=') and (not (self.xri_type == 'i-number' and value[0] == '!')):
            raise Invalid(self.message('noType', state), value, state)
        if self.xri_type == 'i-name':
            self._validate_iname(value, state)
        else:
            self._validate_inumber(value, state)

    def _validate_iname(self, iname, state):
        """Validate an i-name"""
        iname = iname[1:]
        if '..' in iname or '--' in iname:
            raise Invalid(self.message('repeatedChar', state), iname, state)
        if self.iname_invalid_start.match(iname):
            raise Invalid(self.message('badInameStart', state), iname, state)
        if not self.iname_valid_pattern.match(iname) or '_' in iname:
            raise Invalid(self.message('badIname', state, iname=iname), iname, state)

    def _validate_inumber(self, inumber, state):
        """Validate an i-number"""
        if not self.__class__.inumber_pattern.match(inumber):
            raise Invalid(self.message('badInumber', state, inumber=inumber, value=inumber), inumber, state)