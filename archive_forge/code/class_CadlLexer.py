from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Name, Literal, Number, String, \
class CadlLexer(AtomsLexer):
    """
    Lexer for cADL syntax.

    .. versionadded:: 2.1
    """
    name = 'cADL'
    aliases = ['cadl']
    filenames = ['*.cadl']
    tokens = {'path': [('[a-z_]\\w*', Name.Class), ('/', Punctuation), ('\\[', Punctuation, 'any_code'), ('\\s+', Punctuation, '#pop')], 'root': [include('whitespace'), ('(cardinality|existence|occurrences|group|include|exclude|allow_archetype|use_archetype|use_node)\\W', Keyword.Type), ('(and|or|not|there_exists|xor|implies|for_all)\\W', Keyword.Type), ('(after|before|closed)\\W', Keyword.Type), ('(not)\\W', Operator), ('(matches|is_in)\\W', Operator), (u'(∈|∉)', Operator), (u'(∃|∄|∀|∧|∨|⊻|\x93C)', Operator), ('(\\{)(\\s*/[^}]+/\\s*)(\\})', bygroups(Punctuation, String.Regex, Punctuation)), ('(\\{)(\\s*\\^[^}]+\\^\\s*)(\\})', bygroups(Punctuation, String.Regex, Punctuation)), ('/', Punctuation, 'path'), ('(\\{)((?:\\d+\\.\\.)?(?:\\d+|\\*))((?:\\s*;\\s*(?:ordered|unordered|unique)){,2})(\\})', bygroups(Punctuation, Number, Number, Punctuation)), ('\\[\\{', Punctuation), ('\\}\\]', Punctuation), ('\\{', Punctuation), ('\\}', Punctuation), include('constraint_values'), ('[A-Z]\\w+(<[A-Z]\\w+([A-Za-z_<>]*)>)?', Name.Class), ('[a-z_]\\w*', Name.Class), ('\\[', Punctuation, 'any_code'), ('(~|//|\\\\\\\\|\\+|-|/|\\*|\\^|!=|=|<=|>=|<|>]?)', Operator), ('\\(', Punctuation), ('\\)', Punctuation), (',', Punctuation), ('"', String, 'string'), (';', Punctuation)]}