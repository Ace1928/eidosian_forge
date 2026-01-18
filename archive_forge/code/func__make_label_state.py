import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
def _make_label_state(compound, _label=_label, _label_compound=_label_compound, _nl=_nl, _punct=_punct, _string=_string, _variable=_variable):
    state = []
    if compound:
        state.append(('(?=\\))', Text, '#pop'))
    state.append(('(%s?)((?:%s|%s|\\^[%s]?%s|[^"%%^%s%s%s])*)' % (_label_compound if compound else _label, _string, _variable, _nl, '[^)]' if compound else '[\\w\\W]', _nl, _punct, ')' if compound else ''), bygroups(Name.Label, Comment.Single), '#pop'))
    return state