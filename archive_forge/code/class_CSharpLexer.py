import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class CSharpLexer(RegexLexer):
    """
    For `C# <http://msdn2.microsoft.com/en-us/vcsharp/default.aspx>`_
    source code.

    Additional options accepted:

    `unicodelevel`
      Determines which Unicode characters this lexer allows for identifiers.
      The possible values are:

      * ``none`` -- only the ASCII letters and numbers are allowed. This
        is the fastest selection.
      * ``basic`` -- all Unicode characters from the specification except
        category ``Lo`` are allowed.
      * ``full`` -- all Unicode characters as specified in the C# specs
        are allowed.  Note that this means a considerable slowdown since the
        ``Lo`` category has more than 40,000 characters in it!

      The default value is ``basic``.

      .. versionadded:: 0.8
    """
    name = 'C#'
    aliases = ['csharp', 'c#']
    filenames = ['*.cs']
    mimetypes = ['text/x-csharp']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    levels = {'none': '@?[_a-zA-Z]\\w*', 'basic': '@?[_' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl') + ']' + '[' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*', 'full': '@?(?:_|[^' + uni.allexcept('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl') + '])' + '[^' + uni.allexcept('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*'}
    tokens = {}
    token_variants = True
    for levelname, cs_ident in iteritems(levels):
        tokens[levelname] = {'root': [('^([ \\t]*(?:' + cs_ident + '(?:\\[\\])?\\s+)+?)(' + cs_ident + ')(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Punctuation)), ('^\\s*\\[.*?\\]', Name.Attribute), ('[^\\S\\n]+', Text), ('\\\\\\n', Text), ('//.*?\\n', Comment.Single), ('/[*].*?[*]/', Comment.Multiline), ('\\n', Text), ('[~!%^&*()+=|\\[\\]:;,.<>/?-]', Punctuation), ('[{}]', Punctuation), ('@"(""|[^"])*"', String), ('"(\\\\\\\\|\\\\"|[^"\\n])*["\\n]', String), ("'\\\\.'|'[^\\\\]'", String.Char), ('[0-9](\\.[0-9]*)?([eE][+-][0-9]+)?[flFLdD]?|0[xX][0-9a-fA-F]+[Ll]?', Number), ('#[ \\t]*(if|endif|else|elif|define|undef|line|error|warning|region|endregion|pragma)\\b.*?\\n', Comment.Preproc), ('\\b(extern)(\\s+)(alias)\\b', bygroups(Keyword, Text, Keyword)), ('(abstract|as|async|await|base|break|by|case|catch|checked|const|continue|default|delegate|do|else|enum|event|explicit|extern|false|finally|fixed|for|foreach|goto|if|implicit|in|interface|internal|is|let|lock|new|null|on|operator|out|override|params|private|protected|public|readonly|ref|return|sealed|sizeof|stackalloc|static|switch|this|throw|true|try|typeof|unchecked|unsafe|virtual|void|while|get|set|new|partial|yield|add|remove|value|alias|ascending|descending|from|group|into|orderby|select|thenby|where|join|equals)\\b', Keyword), ('(global)(::)', bygroups(Keyword, Punctuation)), ('(bool|byte|char|decimal|double|dynamic|float|int|long|object|sbyte|short|string|uint|ulong|ushort|var)\\b\\??', Keyword.Type), ('(class|struct)(\\s+)', bygroups(Keyword, Text), 'class'), ('(namespace|using)(\\s+)', bygroups(Keyword, Text), 'namespace'), (cs_ident, Name)], 'class': [(cs_ident, Name.Class, '#pop'), default('#pop')], 'namespace': [('(?=\\()', Text, '#pop'), ('(' + cs_ident + '|\\.)+', Name.Namespace, '#pop')]}

    def __init__(self, **options):
        level = get_choice_opt(options, 'unicodelevel', list(self.tokens), 'basic')
        if level not in self._all_tokens:
            self._tokens = self.__class__.process_tokendef(level)
        else:
            self._tokens = self._all_tokens[level]
        RegexLexer.__init__(self, **options)