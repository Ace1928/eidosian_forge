import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class MuPADLexer(RegexLexer):
    """
    A `MuPAD <http://www.mupad.com>`_ lexer.
    Contributed by Christopher Creutzig <christopher@creutzig.de>.

    .. versionadded:: 0.8
    """
    name = 'MuPAD'
    aliases = ['mupad']
    filenames = ['*.mu']
    tokens = {'root': [('//.*?$', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), ('"(?:[^"\\\\]|\\\\.)*"', String), ('\\(|\\)|\\[|\\]|\\{|\\}', Punctuation), ('(?x)\\b(?:\n                next|break|end|\n                axiom|end_axiom|category|end_category|domain|end_domain|inherits|\n                if|%if|then|elif|else|end_if|\n                case|of|do|otherwise|end_case|\n                while|end_while|\n                repeat|until|end_repeat|\n                for|from|to|downto|step|end_for|\n                proc|local|option|save|begin|end_proc|\n                delete|frame\n              )\\b', Keyword), ('(?x)\\b(?:\n                DOM_ARRAY|DOM_BOOL|DOM_COMPLEX|DOM_DOMAIN|DOM_EXEC|DOM_EXPR|\n                DOM_FAIL|DOM_FLOAT|DOM_FRAME|DOM_FUNC_ENV|DOM_HFARRAY|DOM_IDENT|\n                DOM_INT|DOM_INTERVAL|DOM_LIST|DOM_NIL|DOM_NULL|DOM_POLY|DOM_PROC|\n                DOM_PROC_ENV|DOM_RAT|DOM_SET|DOM_STRING|DOM_TABLE|DOM_VAR\n              )\\b', Name.Class), ('(?x)\\b(?:\n                PI|EULER|E|CATALAN|\n                NIL|FAIL|undefined|infinity|\n                TRUE|FALSE|UNKNOWN\n              )\\b', Name.Constant), ('\\b(?:dom|procname)\\b', Name.Builtin.Pseudo), ("\\.|,|:|;|=|\\+|-|\\*|/|\\^|@|>|<|\\$|\\||!|\\'|%|~=", Operator), ('(?x)\\b(?:\n                and|or|not|xor|\n                assuming|\n                div|mod|\n                union|minus|intersect|in|subset\n              )\\b', Operator.Word), ('\\b(?:I|RDN_INF|RD_NINF|RD_NAN)\\b', Number), ('(?x)\n              ((?:[a-zA-Z_#][\\w#]*|`[^`]*`)\n              (?:::[a-zA-Z_#][\\w#]*|`[^`]*`)*)(\\s*)([(])', bygroups(Name.Function, Text, Punctuation)), ('(?x)\n              (?:[a-zA-Z_#][\\w#]*|`[^`]*`)\n              (?:::[a-zA-Z_#][\\w#]*|`[^`]*`)*', Name.Variable), ('[0-9]+(?:\\.[0-9]*)?(?:e[0-9]+)?', Number), ('\\.[0-9]+(?:e[0-9]+)?', Number), ('.', Text)], 'comment': [('[^*/]', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}