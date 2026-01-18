import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def add_default_node_attr(self, **kwds):
    self.default_node_attr.update(kwds)