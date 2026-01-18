import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def PostalCode(*kw, **kwargs):
    deprecation_warning('please use formencode.national.USPostalCode')
    from formencode.national import USPostalCode
    return USPostalCode(*kw, **kwargs)