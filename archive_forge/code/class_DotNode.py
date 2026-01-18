import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotNode(object):
    """Class representing a DOT node"""

    def __init__(self, name, **kwds):
        """Create a Node instance

        Input:
            name - name of node. Have to be a string
            **kwds node attributes

        """
        self.name = name
        self.attr = {}
        self.parent = None
        self.attr.update(kwds)

    def __str__(self):
        attrstr = ','.join(['%s=%s' % (quote_if_necessary(key), quote_if_necessary(val)) for key, val in self.attr.items()])
        if attrstr:
            attrstr = '[%s]' % attrstr
        return '%s%s;\n' % (quote_if_necessary(self.name), attrstr)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        try:
            return self.name == other
        except:
            return False

    def __ne__(self, other):
        try:
            return self.name != other
        except:
            return False

    def __getattr__(self, name):
        try:
            return self.attr[name]
        except KeyError:
            raise AttributeError