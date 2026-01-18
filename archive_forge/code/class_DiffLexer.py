import re
from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, Generic, \
class DiffLexer(RegexLexer):
    """
    Lexer for unified or context-style diffs or patches.
    """
    name = 'Diff'
    aliases = ['diff', 'udiff']
    filenames = ['*.diff', '*.patch']
    mimetypes = ['text/x-diff', 'text/x-patch']
    tokens = {'root': [(' .*\\n', Text), ('\\+.*\\n', Generic.Inserted), ('-.*\\n', Generic.Deleted), ('!.*\\n', Generic.Strong), ('@.*\\n', Generic.Subheading), ('([Ii]ndex|diff).*\\n', Generic.Heading), ('=.*\\n', Generic.Heading), ('.*\\n', Text)]}

    def analyse_text(text):
        if text[:7] == 'Index: ':
            return True
        if text[:5] == 'diff ':
            return True
        if text[:4] == '--- ':
            return 0.9