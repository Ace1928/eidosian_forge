import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
def _make_arithmetic_state(compound, _nl=_nl, _punct=_punct, _string=_string, _variable=_variable, _ws=_ws):
    op = '=+\\-*/!~'
    state = []
    if compound:
        state.append(('(?=\\))', Text, '#pop'))
    state += [('0[0-7]+', Number.Oct), ('0x[\\da-f]+', Number.Hex), ('\\d+', Number.Integer), ('[(),]+', Punctuation), ('([%s]|%%|\\^\\^)+' % op, Operator), ('(%s|%s|(\\^[%s]?)?[^()%s%%^"%s%s%s]|\\^[%s%s]?%s)+' % (_string, _variable, _nl, op, _nl, _punct, _ws, _nl, _ws, '[^)]' if compound else '[\\w\\W]'), using(this, state='variable')), ('(?=[\\x00|&])', Text, '#pop'), include('follow')]
    return state