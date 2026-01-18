from pygments.lexer import RegexLexer, bygroups, include
from pygments.token import Whitespace, Keyword, Name, String, Number, \
class SaviLexer(RegexLexer):
    """
  For Savi source code.

  .. versionadded: 2.10
  """
    name = 'Savi'
    url = 'https://github.com/savi-lang/savi'
    aliases = ['savi']
    filenames = ['*.savi']
    tokens = {'root': [('//.*?$', Comment.Single), ('::.*?$', Comment.Single), ("(\\')(\\w+)(?=[^\\'])", bygroups(Operator, Name)), ('\\w?"', String.Double, 'string.double'), ("'", String.Char, 'string.char'), ('(_?[A-Z]\\w*)', Name.Class), ('(\\.)(\\s*)(_?[A-Z]\\w*)', bygroups(Punctuation, Whitespace, Name.Class)), ('^([ \\t]*)(:\\w+)', bygroups(Whitespace, Name.Tag), 'decl'), ('((\\w+|\\+|\\-|\\*)\\!)', Generic.Deleted), ('\\b\\d([\\d_]*(\\.[\\d_]+)?)\\b', Number), ('\\b0x([0-9a-fA-F_]+)\\b', Number.Hex), ('\\b0b([01_]+)\\b', Number.Bin), ('\\w+(?=\\()', Name.Function), ('(\\.)(\\s*)(\\w+)', bygroups(Punctuation, Whitespace, Name.Function)), ('(@)(\\w+)', bygroups(Punctuation, Name.Function)), ('\\(', Punctuation, 'root'), ('\\)', Punctuation, '#pop'), ('\\{', Punctuation, 'root'), ('\\}', Punctuation, '#pop'), ('\\[', Punctuation, 'root'), ('(\\])(\\!)', bygroups(Punctuation, Generic.Deleted), '#pop'), ('\\]', Punctuation, '#pop'), ('[,;:\\.@]', Punctuation), ('(\\|\\>)', Operator), ('(\\&\\&|\\|\\||\\?\\?|\\&\\?|\\|\\?|\\.\\?)', Operator), ('(\\<\\=\\>|\\=\\~|\\=\\=|\\<\\=|\\>\\=|\\<|\\>)', Operator), ('(\\+|\\-|\\/|\\*|\\%)', Operator), ('(\\=)', Operator), ('(\\!|\\<\\<|\\<|\\&|\\|)', Operator), ('\\b\\w+\\b', Name), ('[ \\t\\r]+\\n*|\\n+', Whitespace)], 'decl': [('\\b[a-z_]\\w*\\b(?!\\!)', Keyword.Declaration), (':', Punctuation, '#pop'), ('\\n', Whitespace, '#pop'), include('root')], 'string.double': [('\\\\\\(', String.Interpol, 'string.interpolation'), ('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape), ("\\\\[bfnrt\\\\\\']", String.Escape), ('\\\\"', String.Escape), ('"', String.Double, '#pop'), ('[^\\\\"]+', String.Double), ('.', Error)], 'string.char': [('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\x[0-9a-fA-F]{2}', String.Escape), ("\\\\[bfnrt\\\\\\']", String.Escape), ("\\\\'", String.Escape), ("'", String.Char, '#pop'), ("[^\\\\']+", String.Char), ('.', Error)], 'string.interpolation': [('\\)', String.Interpol, '#pop'), include('root')]}