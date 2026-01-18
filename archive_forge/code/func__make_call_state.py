import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
def _make_call_state(compound, _label=_label, _label_compound=_label_compound):
    state = []
    if compound:
        state.append(('(?=\\))', Text, '#pop'))
    state.append(('(:?)(%s)' % (_label_compound if compound else _label), bygroups(Punctuation, Name.Label), '#pop'))
    return state