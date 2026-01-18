import cgi
import re
import warnings
from encodings import idna
from .api import (FancyValidator, Identity, Invalid, NoDefault, Validator,
def __initargs__(self, new_attrs):
    if self.not_empty is None and self.min:
        self.not_empty = True