import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class XmlLexer(RegexLexer):
    """
    Generic lexer for XML (eXtensible Markup Language).
    """
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    name = 'XML'
    aliases = ['xml']
    filenames = ['*.xml', '*.xsl', '*.rss', '*.xslt', '*.xsd', '*.wsdl', '*.wsf']
    mimetypes = ['text/xml', 'application/xml', 'image/svg+xml', 'application/rss+xml', 'application/atom+xml']
    tokens = {'root': [('[^<&]+', Text), ('&\\S*?;', Name.Entity), ('\\<\\!\\[CDATA\\[.*?\\]\\]\\>', Comment.Preproc), ('<!--', Comment, 'comment'), ('<\\?.*?\\?>', Comment.Preproc), ('<![^>]*>', Comment.Preproc), ('<\\s*[\\w:.-]+', Name.Tag, 'tag'), ('<\\s*/\\s*[\\w:.-]+\\s*>', Name.Tag)], 'comment': [('[^-]+', Comment), ('-->', Comment, '#pop'), ('-', Comment)], 'tag': [('\\s+', Text), ('[\\w.:-]+\\s*=', Name.Attribute, 'attr'), ('/?\\s*>', Name.Tag, '#pop')], 'attr': [('\\s+', Text), ('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}

    def analyse_text(text):
        if looks_like_xml(text):
            return 0.45