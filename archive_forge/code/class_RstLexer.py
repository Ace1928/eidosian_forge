import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class RstLexer(RegexLexer):
    """
    For `reStructuredText <http://docutils.sf.net/rst.html>`_ markup.

    .. versionadded:: 0.7

    Additional options accepted:

    `handlecodeblocks`
        Highlight the contents of ``.. sourcecode:: language``,
        ``.. code:: language`` and ``.. code-block:: language``
        directives with a lexer for the given language (default:
        ``True``).

        .. versionadded:: 0.8
    """
    name = 'reStructuredText'
    aliases = ['rst', 'rest', 'restructuredtext']
    filenames = ['*.rst', '*.rest']
    mimetypes = ['text/x-rst', 'text/prs.fallenstein.rst']
    flags = re.MULTILINE

    def _handle_sourcecode(self, match):
        from pygments.lexers import get_lexer_by_name
        yield (match.start(1), Punctuation, match.group(1))
        yield (match.start(2), Text, match.group(2))
        yield (match.start(3), Operator.Word, match.group(3))
        yield (match.start(4), Punctuation, match.group(4))
        yield (match.start(5), Text, match.group(5))
        yield (match.start(6), Keyword, match.group(6))
        yield (match.start(7), Text, match.group(7))
        lexer = None
        if self.handlecodeblocks:
            try:
                lexer = get_lexer_by_name(match.group(6).strip())
            except ClassNotFound:
                pass
        indention = match.group(8)
        indention_size = len(indention)
        code = indention + match.group(9) + match.group(10) + match.group(11)
        if lexer is None:
            yield (match.start(8), String, code)
            return
        ins = []
        codelines = code.splitlines(True)
        code = ''
        for line in codelines:
            if len(line) > indention_size:
                ins.append((len(code), [(0, Text, line[:indention_size])]))
                code += line[indention_size:]
            else:
                code += line
        for item in do_insertions(ins, lexer.get_tokens_unprocessed(code)):
            yield item
    closers = u'\'")]}>’”»!?'
    unicode_delimiters = u'‐‑‒–—\xa0'
    end_string_suffix = '((?=$)|(?=[-/:.,; \\n\\x00%s%s]))' % (re.escape(unicode_delimiters), re.escape(closers))
    tokens = {'root': [('^(=+|-+|`+|:+|\\.+|\\\'+|"+|~+|\\^+|_+|\\*+|\\++|#+)([ \\t]*\\n)(.+)(\\n)(\\1)(\\n)', bygroups(Generic.Heading, Text, Generic.Heading, Text, Generic.Heading, Text)), ('^(\\S.*)(\\n)(={3,}|-{3,}|`{3,}|:{3,}|\\.{3,}|\\\'{3,}|"{3,}|~{3,}|\\^{3,}|_{3,}|\\*{3,}|\\+{3,}|#{3,})(\\n)', bygroups(Generic.Heading, Text, Generic.Heading, Text)), ('^(\\s*)([-*+])( .+\\n(?:\\1  .+\\n)*)', bygroups(Text, Number, using(this, state='inline'))), ('^(\\s*)([0-9#ivxlcmIVXLCM]+\\.)( .+\\n(?:\\1  .+\\n)*)', bygroups(Text, Number, using(this, state='inline'))), ('^(\\s*)(\\(?[0-9#ivxlcmIVXLCM]+\\))( .+\\n(?:\\1  .+\\n)*)', bygroups(Text, Number, using(this, state='inline'))), ('^(\\s*)([A-Z]+\\.)( .+\\n(?:\\1  .+\\n)+)', bygroups(Text, Number, using(this, state='inline'))), ('^(\\s*)(\\(?[A-Za-z]+\\))( .+\\n(?:\\1  .+\\n)+)', bygroups(Text, Number, using(this, state='inline'))), ('^(\\s*)(\\|)( .+\\n(?:\\|  .+\\n)*)', bygroups(Text, Operator, using(this, state='inline'))), ('^( *\\.\\.)(\\s*)((?:source)?code(?:-block)?)(::)([ \\t]*)([^\\n]+)(\\n[ \\t]*\\n)([ \\t]+)(.*)(\\n)((?:(?:\\8.*|)\\n)+)', _handle_sourcecode), ('^( *\\.\\.)(\\s*)([\\w:-]+?)(::)(?:([ \\t]*)(.*))', bygroups(Punctuation, Text, Operator.Word, Punctuation, Text, using(this, state='inline'))), ('^( *\\.\\.)(\\s*)(_(?:[^:\\\\]|\\\\.)+:)(.*?)$', bygroups(Punctuation, Text, Name.Tag, using(this, state='inline'))), ('^( *\\.\\.)(\\s*)(\\[.+\\])(.*?)$', bygroups(Punctuation, Text, Name.Tag, using(this, state='inline'))), ('^( *\\.\\.)(\\s*)(\\|.+\\|)(\\s*)([\\w:-]+?)(::)(?:([ \\t]*)(.*))', bygroups(Punctuation, Text, Name.Tag, Text, Operator.Word, Punctuation, Text, using(this, state='inline'))), ('^ *\\.\\..*(\\n( +.*\\n|\\n)+)?', Comment.Preproc), ('^( *)(:[a-zA-Z-]+:)(\\s*)$', bygroups(Text, Name.Class, Text)), ('^( *)(:.*?:)([ \\t]+)(.*?)$', bygroups(Text, Name.Class, Text, Name.Function)), ('^(\\S.*(?<!::)\\n)((?:(?: +.*)\\n)+)', bygroups(using(this, state='inline'), using(this, state='inline'))), ('(::)(\\n[ \\t]*\\n)([ \\t]+)(.*)(\\n)((?:(?:\\3.*|)\\n)+)', bygroups(String.Escape, Text, String, String, Text, String)), include('inline')], 'inline': [('\\\\.', Text), ('``', String, 'literal'), ('(`.+?)(<.+?>)(`__?)', bygroups(String, String.Interpol, String)), ('`.+?`__?', String), ('(`.+?`)(:[a-zA-Z0-9:-]+?:)?', bygroups(Name.Variable, Name.Attribute)), ('(:[a-zA-Z0-9:-]+?:)(`.+?`)', bygroups(Name.Attribute, Name.Variable)), ('\\*\\*.+?\\*\\*', Generic.Strong), ('\\*.+?\\*', Generic.Emph), ('\\[.*?\\]_', String), ('<.+?>', Name.Tag), ('[^\\\\\\n\\[*`:]+', Text), ('.', Text)], 'literal': [('[^`]+', String), ('``' + end_string_suffix, String, '#pop'), ('`', String)]}

    def __init__(self, **options):
        self.handlecodeblocks = get_bool_opt(options, 'handlecodeblocks', True)
        RegexLexer.__init__(self, **options)

    def analyse_text(text):
        if text[:2] == '..' and text[2:3] != '.':
            return 0.3
        p1 = text.find('\n')
        p2 = text.find('\n', p1 + 1)
        if p2 > -1 and p1 * 2 + 1 == p2 and (text[p1 + 1] in '-=') and (text[p1 + 1] == text[p2 - 1]):
            return 0.5