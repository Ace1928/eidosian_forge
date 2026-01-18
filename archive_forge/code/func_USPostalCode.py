import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def USPostalCode(*args, **kw):
    """
    US Postal codes (aka Zip Codes).

    ::

        >>> uspc = USPostalCode()
        >>> uspc.to_python('55555')
        '55555'
        >>> uspc.to_python('55555-5555')
        '55555-5555'
        >>> uspc.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (5 digits)
    """
    return Any(DelimitedDigitsPostalCode(5, None, *args, **kw), DelimitedDigitsPostalCode([5, 4], '-', *args, **kw))