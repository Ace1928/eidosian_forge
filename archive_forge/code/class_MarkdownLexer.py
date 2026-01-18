import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MarkdownLexer(RegexLexer):
    """
    For `Markdown <https://help.github.com/categories/writing-on-github/>`_ markup.

    .. versionadded:: 2.2
    """
    name = 'markdown'
    aliases = ['md']
    filenames = ['*.md']
    mimetypes = ['text/x-markdown']
    flags = re.MULTILINE

    def _handle_codeblock(self, match):
        """
        match args: 1:backticks, 2:lang_name, 3:newline, 4:code, 5:backticks
        """
        from pygments.lexers import get_lexer_by_name
        yield (match.start(1), String, match.group(1))
        yield (match.start(2), String, match.group(2))
        yield (match.start(3), Text, match.group(3))
        lexer = None
        if self.handlecodeblocks:
            try:
                lexer = get_lexer_by_name(match.group(2).strip())
            except ClassNotFound:
                pass
        code = match.group(4)
        if lexer is None:
            yield (match.start(4), String, code)
            return
        for item in do_insertions([], lexer.get_tokens_unprocessed(code)):
            yield item
        yield (match.start(5), String, match.group(5))
    tokens = {'root': [('^(#)([^#].+\\n)', bygroups(Generic.Heading, Text)), ('^(#{2,6})(.+\\n)', bygroups(Generic.Subheading, Text)), ('^(\\s*)([*-] )(\\[[ xX]\\])( .+\\n)', bygroups(Text, Keyword, Keyword, using(this, state='inline'))), ('^(\\s*)([*-])(\\s)(.+\\n)', bygroups(Text, Keyword, Text, using(this, state='inline'))), ('^(\\s*)([0-9]+\\.)( .+\\n)', bygroups(Text, Keyword, using(this, state='inline'))), ('^(\\s*>\\s)(.+\\n)', bygroups(Keyword, Generic.Emph)), ('^(```\\n)([\\w\\W]*?)(^```$)', bygroups(String, Text, String)), ('^(```)(\\w+)(\\n)([\\w\\W]*?)(^```$)', _handle_codeblock), include('inline')], 'inline': [('\\\\.', Text), ('(\\s)([*_][^*_]+[*_])(\\W|\\n)', bygroups(Text, Generic.Emph, Text)), ('(\\s)((\\*\\*|__).*\\3)((?=\\W|\\n))', bygroups(Text, Generic.Strong, None, Text)), ('(\\s)(~~[^~]+~~)((?=\\W|\\n))', bygroups(Text, Generic.Deleted, Text)), ('`[^`]+`', String.Backtick), ('[@#][\\w/:]+', Name.Entity), ('(!?\\[)([^]]+)(\\])(\\()([^)]+)(\\))', bygroups(Text, Name.Tag, Text, Text, Name.Attribute, Text)), ('[^\\\\\\s]+', Text), ('.', Text)]}

    def __init__(self, **options):
        self.handlecodeblocks = get_bool_opt(options, 'handlecodeblocks', True)
        RegexLexer.__init__(self, **options)