from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
@fix_unicode_literals_in_doctest
def _find_closing_brace(string):
    """
    >>> _find_closing_brace('')
    (u'', u'')
    >>> _find_closing_brace('no braces')
    (u'no braces', u'')
    >>> _find_closing_brace('brace at the end}')
    (u'brace at the end}', u'')
    >>> _find_closing_brace('two closing braces}}')
    (u'two closing braces}', u'}')
    >>> _find_closing_brace('two closing} braces} and some text')
    (u'two closing}', u' braces} and some text')
    >>> _find_closing_brace('more {nested{}}{braces}} and the rest}')
    (u'more {nested{}}{braces}}', u' and the rest}')
    """
    up_to_brace = []
    brace_level = 1
    while brace_level >= 1:
        next_brace = BRACE_RE.search(string)
        if not next_brace:
            break
        up_to_brace.append(string[:next_brace.end()])
        string = string[next_brace.end():]
        if next_brace.group() == '{':
            brace_level += 1
        elif next_brace.group() == '}':
            brace_level -= 1
        else:
            raise ValueError(next_brace.group())
    if not up_to_brace:
        up_to_brace, string = ([string], '')
    return (''.join(up_to_brace), string)