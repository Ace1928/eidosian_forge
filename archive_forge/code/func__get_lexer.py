import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
def _get_lexer(self, lang):
    if lang.lower() == 'sql':
        return get_lexer_by_name('postgresql', **self.options)
    tries = [lang]
    if lang.startswith('pl'):
        tries.append(lang[2:])
    if lang.endswith('u'):
        tries.append(lang[:-1])
    if lang.startswith('pl') and lang.endswith('u'):
        tries.append(lang[2:-1])
    for l in tries:
        try:
            return get_lexer_by_name(l, **self.options)
        except ClassNotFound:
            pass
    else:
        return None