import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def PhoneNumber(*kw, **kwargs):
    deprecation_warning('please use formencode.national.USPhoneNumber')
    from formencode.national import USPhoneNumber
    return USPhoneNumber(*kw, **kwargs)