from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
class InvalidCodepointContext(IDNAError):
    """ Exception when the codepoint is not valid in the context it is used """
    pass