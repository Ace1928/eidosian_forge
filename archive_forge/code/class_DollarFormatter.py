import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
class DollarFormatter(FullEvalFormatter):
    """Formatter allowing Itpl style $foo replacement, for names and attribute
    access only. Standard {foo} replacement also works, and allows full
    evaluation of its arguments.

    Examples
    --------
    ::

        In [1]: f = DollarFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format('23 * 76 is $result', result=23*76)
        Out[3]: '23 * 76 is 1748'

        In [4]: f.format('$a or {b}', a=1, b=2)
        Out[4]: '1 or 2'
    """
    _dollar_pattern_ignore_single_quote = re.compile("(.*?)\\$(\\$?[\\w\\.]+)(?=([^']*'[^']*')*[^']*$)")

    def parse(self, fmt_string):
        for literal_txt, field_name, format_spec, conversion in Formatter.parse(self, fmt_string):
            continue_from = 0
            txt = ''
            for m in self._dollar_pattern_ignore_single_quote.finditer(literal_txt):
                new_txt, new_field = m.group(1, 2)
                if new_field.startswith('$'):
                    txt += new_txt + new_field
                else:
                    yield (txt + new_txt, new_field, '', None)
                    txt = ''
                continue_from = m.end()
            yield (txt + literal_txt[continue_from:], field_name, format_spec, conversion)

    def __repr__(self):
        return '<DollarFormatter>'