import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotDefaultAttr(object):

    def __init__(self, element_type, **kwds):
        self.element_type = element_type
        self.attr = kwds

    def __str__(self):
        attrstr = ','.join(['%s=%s' % (quote_if_necessary(key), quote_if_necessary(val)) for key, val in self.attr.items()])
        if attrstr:
            attrstr = '[%s]' % attrstr
            return '%s%s;\n' % (self.element_type, attrstr)
        else:
            return ''