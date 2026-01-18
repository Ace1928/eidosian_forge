import os
from io import open
import types
from functools import wraps, partial
from contextlib import contextmanager
import sys, re
import sre_parse
import sre_constants
from inspect import getmembers, getmro
from functools import partial, wraps
from itertools import repeat, product
class LALR_TraditionalLexer(LALR_WithLexer):

    def init_lexer(self):
        self.init_traditional_lexer()