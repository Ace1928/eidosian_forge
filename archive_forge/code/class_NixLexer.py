import re
from pygments.lexer import RegexLexer, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class NixLexer(RegexLexer):
    """
    For the `Nix language <http://nixos.org/nix/>`_.

    .. versionadded:: 2.0
    """
    name = 'Nix'
    aliases = ['nixos', 'nix']
    filenames = ['*.nix']
    mimetypes = ['text/x-nix']
    flags = re.MULTILINE | re.UNICODE
    keywords = ['rec', 'with', 'let', 'in', 'inherit', 'assert', 'if', 'else', 'then', '...']
    builtins = ['import', 'abort', 'baseNameOf', 'dirOf', 'isNull', 'builtins', 'map', 'removeAttrs', 'throw', 'toString', 'derivation']
    operators = ['++', '+', '?', '.', '!', '//', '==', '!=', '&&', '||', '->', '=']
    punctuations = ['(', ')', '[', ']', ';', '{', '}', ':', ',', '@']
    tokens = {'root': [('#.*$', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), ('\\s+', Text), ('(%s)' % '|'.join((re.escape(entry) + '\\b' for entry in keywords)), Keyword), ('(%s)' % '|'.join((re.escape(entry) + '\\b' for entry in builtins)), Name.Builtin), ('\\b(true|false|null)\\b', Name.Constant), ('(%s)' % '|'.join((re.escape(entry) for entry in operators)), Operator), ('\\b(or|and)\\b', Operator.Word), ('(%s)' % '|'.join((re.escape(entry) for entry in punctuations)), Punctuation), ('[0-9]+', Number.Integer), ('"', String.Double, 'doublequote'), ("''", String.Single, 'singlequote'), ('[\\w.+-]*(\\/[\\w.+-]+)+', Literal), ('\\<[\\w.+-]+(\\/[\\w.+-]+)*\\>', Literal), ("[a-zA-Z][a-zA-Z0-9\\+\\-\\.]*\\:[\\w%/?:@&=+$,\\\\.!~*\\'-]+", Literal), ('[\\w-]+\\s*=', String.Symbol), ("[a-zA-Z_][\\w\\'-]*", Text)], 'comment': [('[^/*]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)], 'singlequote': [("'''", String.Escape), ("''\\$\\{", String.Escape), ("''\\n", String.Escape), ("''\\r", String.Escape), ("''\\t", String.Escape), ("''", String.Single, '#pop'), ('\\$\\{', String.Interpol, 'antiquote'), ("[^']", String.Single)], 'doublequote': [('\\\\', String.Escape), ('\\\\"', String.Escape), ('\\\\$\\{', String.Escape), ('"', String.Double, '#pop'), ('\\$\\{', String.Interpol, 'antiquote'), ('[^"]', String.Double)], 'antiquote': [('\\}', String.Interpol, '#pop'), ('\\$\\{', String.Interpol, '#push'), include('root')]}

    def analyse_text(text):
        rv = 0.0
        if re.search('import.+?<[^>]+>', text):
            rv += 0.4
        if re.search('mkDerivation\\s+(\\(|\\{|rec)', text):
            rv += 0.4
        if re.search('=\\s+mkIf\\s+', text):
            rv += 0.4
        if re.search('\\{[a-zA-Z,\\s]+\\}:', text):
            rv += 0.1
        return rv