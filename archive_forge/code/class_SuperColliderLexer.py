import re
from pygments.lexer import RegexLexer, include, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SuperColliderLexer(RegexLexer):
    """
    For `SuperCollider <http://supercollider.github.io/>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'SuperCollider'
    aliases = ['sc', 'supercollider']
    filenames = ['*.sc', '*.scd']
    mimetypes = ['application/supercollider', 'text/supercollider']
    flags = re.DOTALL | re.MULTILINE
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('<!--', Comment), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [('^(?=\\s|/|<!--)', Text, 'slashstartsregex'), include('commentsandwhitespace'), ('\\+\\+|--|~|&&|\\?|:|\\|\\||\\\\(?=\\n)|(<<|>>>?|==?|!=?|[-<>+*%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), (words(('for', 'in', 'while', 'do', 'break', 'return', 'continue', 'switch', 'case', 'default', 'if', 'else', 'throw', 'try', 'catch', 'finally', 'new', 'delete', 'typeof', 'instanceof', 'void'), suffix='\\b'), Keyword, 'slashstartsregex'), (words(('var', 'let', 'with', 'function', 'arg'), suffix='\\b'), Keyword.Declaration, 'slashstartsregex'), (words(('(abstract', 'boolean', 'byte', 'char', 'class', 'const', 'debugger', 'double', 'enum', 'export', 'extends', 'final', 'float', 'goto', 'implements', 'import', 'int', 'interface', 'long', 'native', 'package', 'private', 'protected', 'public', 'short', 'static', 'super', 'synchronized', 'throws', 'transient', 'volatile'), suffix='\\b'), Keyword.Reserved), (words(('true', 'false', 'nil', 'inf'), suffix='\\b'), Keyword.Constant), (words(('Array', 'Boolean', 'Date', 'Error', 'Function', 'Number', 'Object', 'Packages', 'RegExp', 'String', 'isFinite', 'isNaN', 'parseFloat', 'parseInt', 'super', 'thisFunctionDef', 'thisFunction', 'thisMethod', 'thisProcess', 'thisThread', 'this'), suffix='\\b'), Name.Builtin), ('[$a-zA-Z_]\\w*', Name.Other), ('\\\\?[$a-zA-Z_]\\w*', String.Symbol), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"(\\\\\\\\|\\\\"|[^"])*"', String.Double), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single)]}