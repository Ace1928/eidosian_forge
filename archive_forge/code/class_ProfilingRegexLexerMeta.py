from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
class ProfilingRegexLexerMeta(RegexLexerMeta):
    """Metaclass for ProfilingRegexLexer, collects regex timing info."""

    def _process_regex(cls, regex, rflags, state):
        if isinstance(regex, words):
            rex = regex_opt(regex.words, prefix=regex.prefix, suffix=regex.suffix)
        else:
            rex = regex
        compiled = re.compile(rex, rflags)

        def match_func(text, pos, endpos=sys.maxsize):
            info = cls._prof_data[-1].setdefault((state, rex), [0, 0.0])
            t0 = time.time()
            res = compiled.match(text, pos, endpos)
            t1 = time.time()
            info[0] += 1
            info[1] += t1 - t0
            return res
        return match_func