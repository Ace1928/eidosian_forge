import re
from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, Generic, \
class DarcsPatchLexer(RegexLexer):
    """
    DarcsPatchLexer is a lexer for the various versions of the darcs patch
    format.  Examples of this format are derived by commands such as
    ``darcs annotate --patch`` and ``darcs send``.

    .. versionadded:: 0.10
    """
    name = 'Darcs Patch'
    aliases = ['dpatch']
    filenames = ['*.dpatch', '*.darcspatch']
    DPATCH_KEYWORDS = ('hunk', 'addfile', 'adddir', 'rmfile', 'rmdir', 'move', 'replace')
    tokens = {'root': [('<', Operator), ('>', Operator), ('\\{', Operator), ('\\}', Operator), ('(\\[)((?:TAG )?)(.*)(\\n)(.*)(\\*\\*)(\\d+)(\\s?)(\\])', bygroups(Operator, Keyword, Name, Text, Name, Operator, Literal.Date, Text, Operator)), ('(\\[)((?:TAG )?)(.*)(\\n)(.*)(\\*\\*)(\\d+)(\\s?)', bygroups(Operator, Keyword, Name, Text, Name, Operator, Literal.Date, Text), 'comment'), ('New patches:', Generic.Heading), ('Context:', Generic.Heading), ('Patch bundle hash:', Generic.Heading), ('(\\s*)(%s)(.*\\n)' % '|'.join(DPATCH_KEYWORDS), bygroups(Text, Keyword, Text)), ('\\+', Generic.Inserted, 'insert'), ('-', Generic.Deleted, 'delete'), ('.*\\n', Text)], 'comment': [('[^\\]].*\\n', Comment), ('\\]', Operator, '#pop')], 'specialText': [('\\n', Text, '#pop'), ('\\[_[^_]*_]', Operator)], 'insert': [include('specialText'), ('\\[', Generic.Inserted), ('[^\\n\\[]+', Generic.Inserted)], 'delete': [include('specialText'), ('\\[', Generic.Deleted), ('[^\\n\\[]+', Generic.Deleted)]}