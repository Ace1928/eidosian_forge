import re
from pygments.lexer import RegexLexer, bygroups, include, this, using, words
from pygments.token import Comment, Keyword, Literal, Name, Number, \
class BnfLexer(RegexLexer):
    """
    This lexer is for grammer notations which are similar to
    original BNF.

    In order to maximize a number of targets of this lexer,
    let's decide some designs:

    * We don't distinguish `Terminal Symbol`.

    * We do assume that `NonTerminal Symbol` are always enclosed
      with arrow brackets.

    * We do assume that `NonTerminal Symbol` may include
      any printable characters except arrow brackets and ASCII 0x20.
      This assumption is for `RBNF <http://www.rfc-base.org/txt/rfc-5511.txt>`_.

    * We do assume that target notation doesn't support comment.

    * We don't distinguish any operators and punctuation except
      `::=`.

    Though these desision making might cause too minimal highlighting
    and you might be disappointed, but it is reasonable for us.

    .. versionadded:: 2.1
    """
    name = 'BNF'
    aliases = ['bnf']
    filenames = ['*.bnf']
    mimetypes = ['text/x-bnf']
    tokens = {'root': [('(<)([ -;=?-~]+)(>)', bygroups(Punctuation, Name.Class, Punctuation)), ('::=', Operator), ('[^<>:]+', Text), ('.', Text)]}