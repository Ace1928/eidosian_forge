import bisect
import sys
import logging
import os
import os.path
import ply.lex as lex
import ply.yacc as yacc
from inspect import getfile, currentframe
from pyomo.common.fileutils import this_file
from pyomo.core.base.util import flatten_tuple
def _lex_token_position(t):
    i = bisect.bisect_left(t.lexer.linepos, t.lexpos)
    if i:
        return t.lexpos - t.lexer.linepos[i - 1]
    return t.lexpos