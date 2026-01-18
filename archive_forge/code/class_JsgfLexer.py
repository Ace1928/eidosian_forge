import re
from pygments.lexer import RegexLexer, bygroups, include, this, using, words
from pygments.token import Comment, Keyword, Literal, Name, Number, \
class JsgfLexer(RegexLexer):
    """
    For `JSpeech Grammar Format <https://www.w3.org/TR/jsgf/>`_
    grammars.

    .. versionadded:: 2.2
    """
    name = 'JSGF'
    aliases = ['jsgf']
    filenames = ['*.jsgf']
    mimetypes = ['application/jsgf', 'application/x-jsgf', 'text/jsgf']
    flags = re.MULTILINE | re.UNICODE
    tokens = {'root': [include('comments'), include('non-comments')], 'comments': [('/\\*\\*(?!/)', Comment.Multiline, 'documentation comment'), ('/\\*[\\w\\W]*?\\*/', Comment.Multiline), ('//.*', Comment.Single)], 'non-comments': [('\\A#JSGF[^;]*', Comment.Preproc), ('\\s+', Text), (';', Punctuation), ('[=|()\\[\\]*+]', Operator), ('/[^/]+/', Number.Float), ('"', String.Double, 'string'), ('\\{', String.Other, 'tag'), (words(('import', 'public'), suffix='\\b'), Keyword.Reserved), ('grammar\\b', Keyword.Reserved, 'grammar name'), ('(<)(NULL|VOID)(>)', bygroups(Punctuation, Name.Builtin, Punctuation)), ('<', Punctuation, 'rulename'), ('\\w+|[^\\s;=|()\\[\\]*+/"{<\\w]+', Text)], 'string': [('"', String.Double, '#pop'), ('\\\\.', String.Escape), ('[^\\\\"]+', String.Double)], 'tag': [('\\}', String.Other, '#pop'), ('\\\\.', String.Escape), ('[^\\\\}]+', String.Other)], 'grammar name': [(';', Punctuation, '#pop'), ('\\s+', Text), ('\\.', Punctuation), ('[^;\\s.]+', Name.Namespace)], 'rulename': [('>', Punctuation, '#pop'), ('\\*', Punctuation), ('\\s+', Text), ('([^.>]+)(\\s*)(\\.)', bygroups(Name.Namespace, Text, Punctuation)), ('[^.>]+', Name.Constant)], 'documentation comment': [('\\*/', Comment.Multiline, '#pop'), ('(^\\s*\\*?\\s*)(@(?:example|see)\\s+)([\\w\\W]*?(?=(?:^\\s*\\*?\\s*@|\\*/)))', bygroups(Comment.Multiline, Comment.Special, using(this, state='example'))), ('(^\\s*\\*?\\s*)(@\\S*)', bygroups(Comment.Multiline, Comment.Special)), ('[^*\\n@]+|\\w|\\W', Comment.Multiline)], 'example': [('\\n\\s*\\*', Comment.Multiline), include('non-comments'), ('.', Comment.Multiline)]}