import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def _steal_trailing_WSP_if_exists(lines):
    wsp = ''
    if lines and lines[-1] and (lines[-1][-1] in WSP):
        wsp = lines[-1][-1]
        lines[-1] = lines[-1][:-1]
    return wsp