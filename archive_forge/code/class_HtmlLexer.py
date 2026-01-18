import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class HtmlLexer(RegexLexer):
    """
    For HTML 4 and XHTML 1 markup. Nested JavaScript and CSS is highlighted
    by the appropriate lexer.
    """
    name = 'HTML'
    aliases = ['html']
    filenames = ['*.html', '*.htm', '*.xhtml', '*.xslt']
    mimetypes = ['text/html', 'application/xhtml+xml']
    flags = re.IGNORECASE | re.DOTALL
    tokens = {'root': [('[^<&]+', Text), ('&\\S*?;', Name.Entity), ('\\<\\!\\[CDATA\\[.*?\\]\\]\\>', Comment.Preproc), ('<!--', Comment, 'comment'), ('<\\?.*?\\?>', Comment.Preproc), ('<![^>]*>', Comment.Preproc), ('(<)(\\s*)(script)(\\s*)', bygroups(Punctuation, Text, Name.Tag, Text), ('script-content', 'tag')), ('(<)(\\s*)(style)(\\s*)', bygroups(Punctuation, Text, Name.Tag, Text), ('style-content', 'tag')), ('(<)(\\s*)([\\w:.-]+)', bygroups(Punctuation, Text, Name.Tag), 'tag'), ('(<)(\\s*)(/)(\\s*)([\\w:.-]+)(\\s*)(>)', bygroups(Punctuation, Text, Punctuation, Text, Name.Tag, Text, Punctuation))], 'comment': [('[^-]+', Comment), ('-->', Comment, '#pop'), ('-', Comment)], 'tag': [('\\s+', Text), ('([\\w:-]+\\s*)(=)(\\s*)', bygroups(Name.Attribute, Operator, Text), 'attr'), ('[\\w:-]+', Name.Attribute), ('(/?)(\\s*)(>)', bygroups(Punctuation, Text, Punctuation), '#pop')], 'script-content': [('(<)(\\s*)(/)(\\s*)(script)(\\s*)(>)', bygroups(Punctuation, Text, Punctuation, Text, Name.Tag, Text, Punctuation), '#pop'), ('.+?(?=<\\s*/\\s*script\\s*>)', using(JavascriptLexer))], 'style-content': [('(<)(\\s*)(/)(\\s*)(style)(\\s*)(>)', bygroups(Punctuation, Text, Punctuation, Text, Name.Tag, Text, Punctuation), '#pop'), ('.+?(?=<\\s*/\\s*style\\s*>)', using(CssLexer))], 'attr': [('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}

    def analyse_text(text):
        if html_doctype_matches(text):
            return 0.5