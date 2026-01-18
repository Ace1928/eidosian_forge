import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class CountryValidator(FancyValidator):
    """
    Will convert a country's name into its ISO-3166 abbreviation for unified
    storage in databases etc. and return a localized country name in the
    reverse step.

    @See http://www.iso.org/iso/country_codes/iso_3166_code_lists.htm

    ::

        >>> CountryValidator.to_python('Germany')
        'DE'
        >>> CountryValidator.to_python('Finland')
        'FI'
        >>> CountryValidator.to_python('UNITED STATES')
        'US'
        >>> CountryValidator.to_python('Krakovia')
        Traceback (most recent call last):
            ...
        Invalid: That country is not listed in ISO 3166
        >>> CountryValidator.from_python('DE')
        'Germany'
        >>> CountryValidator.from_python('FI')
        'Finland'
    """
    key_ok = True
    messages = dict(valueNotFound=_('That country is not listed in ISO 3166'))

    def __init__(self, *args, **kw):
        FancyValidator.__init__(self, *args, **kw)
        if no_country:
            warnings.warn(no_country, Warning, 2)

    def _convert_to_python(self, value, state):
        upval = value.upper()
        if self.key_ok:
            try:
                get_country(upval)
            except Exception:
                pass
            else:
                return upval
        for k, v in get_countries():
            if v.upper() == upval:
                return k
        raise Invalid(self.message('valueNotFound', state), value, state)

    def _convert_from_python(self, value, state):
        try:
            return get_country(value.upper())
        except KeyError:
            return value