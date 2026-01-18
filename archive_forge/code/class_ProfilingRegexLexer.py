from __future__ import print_function
import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
@add_metaclass(ProfilingRegexLexerMeta)
class ProfilingRegexLexer(RegexLexer):
    """Drop-in replacement for RegexLexer that does profiling of its regexes."""
    _prof_data = []
    _prof_sort_index = 4

    def get_tokens_unprocessed(self, text, stack=('root',)):
        self.__class__._prof_data.append({})
        for tok in RegexLexer.get_tokens_unprocessed(self, text, stack):
            yield tok
        rawdata = self.__class__._prof_data.pop()
        data = sorted(((s, repr(r).strip("u'").replace('\\\\', '\\')[:65], n, 1000 * t, 1000 * t / n) for (s, r), (n, t) in rawdata.items()), key=lambda x: x[self._prof_sort_index], reverse=True)
        sum_total = sum((x[3] for x in data))
        print()
        print('Profiling result for %s lexing %d chars in %.3f ms' % (self.__class__.__name__, len(text), sum_total))
        print('=' * 110)
        print('%-20s %-64s ncalls  tottime  percall' % ('state', 'regex'))
        print('-' * 110)
        for d in data:
            print('%-20s %-65s %5d %8.4f %8.4f' % d)
        print('=' * 110)