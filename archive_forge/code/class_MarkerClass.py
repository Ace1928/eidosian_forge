import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
class MarkerClass(object):

    def __repr__(self):
        return msg