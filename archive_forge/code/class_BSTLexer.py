import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, default, \
from pygments.token import Name, Comment, String, Error, Number, Text, \
class BSTLexer(RegexLexer):
    """
    A lexer for BibTeX bibliography styles.

    .. versionadded:: 2.2
    """
    name = 'BST'
    aliases = ['bst', 'bst-pybtex']
    filenames = ['*.bst']
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [include('whitespace'), (words(['read', 'sort']), Keyword), (words(['execute', 'integers', 'iterate', 'reverse', 'strings']), Keyword, 'group'), (words(['function', 'macro']), Keyword, ('group', 'group')), (words(['entry']), Keyword, ('group', 'group', 'group'))], 'group': [include('whitespace'), ('\\{', Punctuation, ('#pop', 'group-end', 'body'))], 'group-end': [include('whitespace'), ('\\}', Punctuation, '#pop')], 'body': [include('whitespace'), ('\\\'[^#\\"\\{\\}\\s]+', Name.Function), ('[^#\\"\\{\\}\\s]+\\$', Name.Builtin), ('[^#\\"\\{\\}\\s]+', Name.Variable), ('"[^\\"]*"', String), ('#-?\\d+', Number), ('\\{', Punctuation, ('group-end', 'body')), default('#pop')], 'whitespace': [('\\s+', Text), ('%.*?$', Comment.SingleLine)]}