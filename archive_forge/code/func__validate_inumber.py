import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def _validate_inumber(self, inumber, state):
    """Validate an i-number"""
    if not self.__class__.inumber_pattern.match(inumber):
        raise Invalid(self.message('badInumber', state, inumber=inumber, value=inumber), inumber, state)