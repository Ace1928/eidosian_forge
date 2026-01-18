import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def FourDigitsPostalCode(*args, **kw):
    return DelimitedDigitsPostalCode(4, None, *args, **kw)