from itertools import chain
import re
import warnings
from xml.sax.saxutils import unescape
from bleach import html5lib_shim
from bleach import parse_shim
class NoCssSanitizerWarning(UserWarning):
    pass