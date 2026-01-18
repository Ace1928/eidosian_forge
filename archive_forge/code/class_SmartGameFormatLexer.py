from pygments.lexer import RegexLexer, bygroups
from pygments.token import Name, Literal, String, Punctuation, Whitespace
class SmartGameFormatLexer(RegexLexer):
    """
    Lexer for Smart Game Format (sgf) file format.

    The format is used to store game records of board games for two players
    (mainly Go game).

    .. versionadded:: 2.4
    """
    name = 'SmartGameFormat'
    url = 'https://www.red-bean.com/sgf/'
    aliases = ['sgf']
    filenames = ['*.sgf']
    tokens = {'root': [('[():;]+', Punctuation), ('(A[BW]|AE|AN|AP|AR|AS|[BW]L|BM|[BW]R|[BW]S|[BW]T|CA|CH|CP|CR|DD|DM|DO|DT|EL|EV|EX|FF|FG|G[BW]|GC|GM|GN|HA|HO|ID|IP|IT|IY|KM|KO|LB|LN|LT|L|MA|MN|M|N|OB|OM|ON|OP|OT|OV|P[BW]|PC|PL|PM|RE|RG|RO|RU|SO|SC|SE|SI|SL|SO|SQ|ST|SU|SZ|T[BW]|TC|TE|TM|TR|UC|US|VW|V|[BW]|C)', Name.Builtin), ('(\\[)([0-9.]+)(\\])', bygroups(Punctuation, Literal.Number, Punctuation)), ('(\\[)([0-9]{4}-[0-9]{2}-[0-9]{2})(\\])', bygroups(Punctuation, Literal.Date, Punctuation)), ('(\\[)([a-z]{2})(\\])', bygroups(Punctuation, String, Punctuation)), ('(\\[)([a-z]{2})(:)([a-z]{2})(\\])', bygroups(Punctuation, String, Punctuation, String, Punctuation)), ('(\\[)([\\w\\s#()+,\\-.:?]+)(\\])', bygroups(Punctuation, String, Punctuation)), ('(\\[)(\\s.*)(\\])', bygroups(Punctuation, Whitespace, Punctuation)), ('\\s+', Whitespace)]}