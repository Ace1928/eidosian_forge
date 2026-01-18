import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def _proc_subgraph_stmt(self, toks):
    """Returns (ADD_SUBGRAPH, name, elements)"""
    return ('add_subgraph', toks[1], toks[2].asList())