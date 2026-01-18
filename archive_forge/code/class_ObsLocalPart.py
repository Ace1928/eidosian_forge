import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class ObsLocalPart(TokenList):
    token_type = 'obs-local-part'
    as_ew_allowed = False