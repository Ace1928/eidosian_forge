import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, include, \
from pygments.token import Punctuation, \
from pygments.util import get_choice_opt, iteritems
from pygments import unistring as uni
from pygments.lexers.html import XmlLexer
class NemerleLexer(RegexLexer):
    """
    For `Nemerle <http://nemerle.org>`_ source code.

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

    .. versionadded:: 1.5
    """
    name = 'Nemerle'
    aliases = ['nemerle']
    filenames = ['*.n']
    mimetypes = ['text/x-nemerle']
    flags = re.MULTILINE | re.DOTALL | re.UNICODE
    levels = {'none': '@?[_a-zA-Z]\\w*', 'basic': '@?[_' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl') + ']' + '[' + uni.combine('Lu', 'Ll', 'Lt', 'Lm', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*', 'full': '@?(?:_|[^' + uni.allexcept('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl') + '])' + '[^' + uni.allexcept('Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Nl', 'Nd', 'Pc', 'Cf', 'Mn', 'Mc') + ']*'}
    tokens = {}
    token_variants = True
    for levelname, cs_ident in iteritems(levels):
        tokens[levelname] = {'root': [('^([ \\t]*(?:' + cs_ident + '(?:\\[\\])?\\s+)+?)(' + cs_ident + ')(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Punctuation)), ('^\\s*\\[.*?\\]', Name.Attribute), ('[^\\S\\n]+', Text), ('\\\\\\n', Text), ('//.*?\\n', Comment.Single), ('/[*].*?[*]/', Comment.Multiline), ('\\n', Text), ('\\$\\s*"', String, 'splice-string'), ('\\$\\s*<#', String, 'splice-string2'), ('<#', String, 'recursive-string'), ('(<\\[)\\s*(' + cs_ident + ':)?', Keyword), ('\\]\\>', Keyword), ('\\$' + cs_ident, Name), ('(\\$)(\\()', bygroups(Name, Punctuation), 'splice-string-content'), ('[~!%^&*()+=|\\[\\]:;,.<>/?-]', Punctuation), ('[{}]', Punctuation), ('@"(""|[^"])*"', String), ('"(\\\\\\\\|\\\\"|[^"\\n])*["\\n]', String), ("'\\\\.'|'[^\\\\]'", String.Char), ('0[xX][0-9a-fA-F]+[Ll]?', Number), ('[0-9](\\.[0-9]*)?([eE][+-][0-9]+)?[flFLdD]?', Number), ('#[ \\t]*(if|endif|else|elif|define|undef|line|error|warning|region|endregion|pragma)\\b.*?\\n', Comment.Preproc), ('\\b(extern)(\\s+)(alias)\\b', bygroups(Keyword, Text, Keyword)), ('(abstract|and|as|base|catch|def|delegate|enum|event|extern|false|finally|fun|implements|interface|internal|is|macro|match|matches|module|mutable|new|null|out|override|params|partial|private|protected|public|ref|sealed|static|syntax|this|throw|true|try|type|typeof|virtual|volatile|when|where|with|assert|assert2|async|break|checked|continue|do|else|ensures|for|foreach|if|late|lock|new|nolate|otherwise|regexp|repeat|requires|return|surroundwith|unchecked|unless|using|while|yield)\\b', Keyword), ('(global)(::)', bygroups(Keyword, Punctuation)), ('(bool|byte|char|decimal|double|float|int|long|object|sbyte|short|string|uint|ulong|ushort|void|array|list)\\b\\??', Keyword.Type), ('(:>?)\\s*(' + cs_ident + '\\??)', bygroups(Punctuation, Keyword.Type)), ('(class|struct|variant|module)(\\s+)', bygroups(Keyword, Text), 'class'), ('(namespace|using)(\\s+)', bygroups(Keyword, Text), 'namespace'), (cs_ident, Name)], 'class': [(cs_ident, Name.Class, '#pop')], 'namespace': [('(?=\\()', Text, '#pop'), ('(' + cs_ident + '|\\.)+', Name.Namespace, '#pop')], 'splice-string': [('[^"$]', String), ('\\$' + cs_ident, Name), ('(\\$)(\\()', bygroups(Name, Punctuation), 'splice-string-content'), ('\\\\"', String), ('"', String, '#pop')], 'splice-string2': [('[^#<>$]', String), ('\\$' + cs_ident, Name), ('(\\$)(\\()', bygroups(Name, Punctuation), 'splice-string-content'), ('<#', String, '#push'), ('#>', String, '#pop')], 'recursive-string': [('[^#<>]', String), ('<#', String, '#push'), ('#>', String, '#pop')], 'splice-string-content': [('if|match', Keyword), ('[~!%^&*+=|\\[\\]:;,.<>/?-\\\\"$ ]', Punctuation), (cs_ident, Name), ('\\d+', Number), ('\\(', Punctuation, '#push'), ('\\)', Punctuation, '#pop')]}

    def __init__(self, **options):
        level = get_choice_opt(options, 'unicodelevel', list(self.tokens), 'basic')
        if level not in self._all_tokens:
            self._tokens = self.__class__.process_tokendef(level)
        else:
            self._tokens = self._all_tokens[level]
        RegexLexer.__init__(self, **options)