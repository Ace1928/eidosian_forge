import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class SingularityLexer(RegexLexer):
    """
    Lexer for Singularity definition files.

    .. versionadded:: 2.6
    """
    name = 'Singularity'
    url = 'https://www.sylabs.io/guides/3.0/user-guide/definition_files.html'
    aliases = ['singularity']
    filenames = ['*.def', 'Singularity']
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    _headers = '^(\\s*)(bootstrap|from|osversion|mirrorurl|include|registry|namespace|includecmd)(:)'
    _section = '^(%(?:pre|post|setup|environment|help|labels|test|runscript|files|startscript))(\\s*)'
    _appsect = '^(%app(?:install|help|run|labels|env|test|files))(\\s*)'
    tokens = {'root': [(_section, bygroups(Generic.Heading, Whitespace), 'script'), (_appsect, bygroups(Generic.Heading, Whitespace), 'script'), (_headers, bygroups(Whitespace, Keyword, Text)), ('\\s*#.*?\\n', Comment), ('\\b(([0-9]+\\.?[0-9]*)|(\\.[0-9]+))\\b', Number), ('[ \\t]+', Whitespace), ('(?!^\\s*%).', Text)], 'script': [('(.+?(?=^\\s*%))|(.*)', using(BashLexer), '#pop')]}

    def analyse_text(text):
        """This is a quite simple script file, but there are a few keywords
        which seem unique to this language."""
        result = 0
        if re.search('\\b(?:osversion|includecmd|mirrorurl)\\b', text, re.IGNORECASE):
            result += 0.5
        if re.search(SingularityLexer._section[1:], text):
            result += 0.49
        return result