import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
class EWWhiteSpaceTerminal(WhiteSpaceTerminal):

    @property
    def value(self):
        return ''

    def __str__(self):
        return ''