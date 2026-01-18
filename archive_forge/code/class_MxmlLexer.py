import re
from pygments.lexer import RegexLexer, bygroups, using, this, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class MxmlLexer(RegexLexer):
    """
    For MXML markup.
    Nested AS3 in <script> tags is highlighted by the appropriate lexer.

    .. versionadded:: 1.1
    """
    flags = re.MULTILINE | re.DOTALL
    name = 'MXML'
    aliases = ['mxml']
    filenames = ['*.mxml']
    mimetimes = ['text/xml', 'application/xml']
    tokens = {'root': [('[^<&]+', Text), ('&\\S*?;', Name.Entity), ('(\\<\\!\\[CDATA\\[)(.*?)(\\]\\]\\>)', bygroups(String, using(ActionScript3Lexer), String)), ('<!--', Comment, 'comment'), ('<\\?.*?\\?>', Comment.Preproc), ('<![^>]*>', Comment.Preproc), ('<\\s*[\\w:.-]+', Name.Tag, 'tag'), ('<\\s*/\\s*[\\w:.-]+\\s*>', Name.Tag)], 'comment': [('[^-]+', Comment), ('-->', Comment, '#pop'), ('-', Comment)], 'tag': [('\\s+', Text), ('[\\w.:-]+\\s*=', Name.Attribute, 'attr'), ('/?\\s*>', Name.Tag, '#pop')], 'attr': [('\\s+', Text), ('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}