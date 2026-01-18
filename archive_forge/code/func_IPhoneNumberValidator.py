import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def IPhoneNumberValidator(*kw, **kwargs):
    deprecation_warning('please use formencode.national.InternationalPhoneNumber')
    from formencode.national import InternationalPhoneNumber
    return InternationalPhoneNumber(*kw, **kwargs)