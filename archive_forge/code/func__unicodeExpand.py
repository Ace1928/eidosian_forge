import codecs
import re
import warnings
from typing import Match
def _unicodeExpand(s):
    return r_unicodeEscape.sub(lambda m: chr(int(m.group(0)[2:], 16)), s)