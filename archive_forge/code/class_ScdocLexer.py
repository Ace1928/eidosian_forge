import re
from pygments.lexer import RegexLexer, include, bygroups, using, this
from pygments.token import Text, Comment, Keyword, String, Generic
class ScdocLexer(RegexLexer):
    """
    `scdoc` is a simple man page generator for POSIX systems written in C99.

    .. versionadded:: 2.5
    """
    name = 'scdoc'
    url = 'https://git.sr.ht/~sircmpwn/scdoc'
    aliases = ['scdoc', 'scd']
    filenames = ['*.scd', '*.scdoc']
    flags = re.MULTILINE
    tokens = {'root': [('^(;.+\\n)', bygroups(Comment)), ('^(#)([^#].+\\n)', bygroups(Generic.Heading, Text)), ('^(#{2})(.+\\n)', bygroups(Generic.Subheading, Text)), ('^(\\s*)([*-])(\\s)(.+\\n)', bygroups(Text, Keyword, Text, using(this, state='inline'))), ('^(\\s*)(\\.+\\.)( .+\\n)', bygroups(Text, Keyword, using(this, state='inline'))), ('^(\\s*>\\s)(.+\\n)', bygroups(Keyword, Generic.Emph)), ('^(```\\n)([\\w\\W]*?)(^```$)', bygroups(String, Text, String)), include('inline')], 'inline': [('\\\\.', Text), ('(\\s)(_[^_]+_)(\\W|\\n)', bygroups(Text, Generic.Emph, Text)), ('(\\s)(\\*[^*]+\\*)(\\W|\\n)', bygroups(Text, Generic.Strong, Text)), ('`[^`]+`', String.Backtick), ('[^\\\\\\s]+', Text), ('.', Text)]}

    def analyse_text(text):
        """We checks for bold and underline text with * and _. Also
        every scdoc file must start with a strictly defined first line."""
        result = 0
        if '*' in text:
            result += 0.01
        if '_' in text:
            result += 0.01
        first_line = text.partition('\n')[0]
        scdoc_preamble_pattern = '^.*\\([1-7]\\)( "[^"]+"){0,2}$'
        if re.search(scdoc_preamble_pattern, first_line):
            result += 0.5
        return result