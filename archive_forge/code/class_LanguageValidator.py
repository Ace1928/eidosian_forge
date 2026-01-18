import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class LanguageValidator(FancyValidator):
    """
    Converts a given language into its ISO 639 alpha 2 code, if there is any.
    Returns the language's full name in the reverse.

    Warning: ISO 639 neither differentiates between languages such as Cantonese
    and Mandarin nor does it contain all spoken languages. E.g., Lechitic
    languages are missing.
    Warning: ISO 639 is a smaller subset of ISO 639-2

    @param  key_ok  accept the language's code instead of its name for input
                    defaults to True

    ::

        >>> l = LanguageValidator()
        >>> l.to_python('German')
        'de'
        >>> l.to_python('Chinese')
        'zh'
        >>> l.to_python('Klingonian')
        Traceback (most recent call last):
            ...
        Invalid: That language is not listed in ISO 639
        >>> l.from_python('de')
        'German'
        >>> l.from_python('zh')
        'Chinese'
    """
    key_ok = True
    messages = dict(valueNotFound=_('That language is not listed in ISO 639'))

    def __init__(self, *args, **kw):
        FancyValidator.__init__(self, *args, **kw)
        if no_country:
            warnings.warn(no_country, Warning, 2)

    def _convert_to_python(self, value, state):
        upval = value.upper()
        if self.key_ok:
            try:
                get_language(value)
            except Exception:
                pass
            else:
                return value
        for k, v in get_languages():
            if v.upper() == upval:
                return k
        raise Invalid(self.message('valueNotFound', state), value, state)

    def _convert_from_python(self, value, state):
        try:
            return get_language(value.lower())
        except KeyError:
            return value