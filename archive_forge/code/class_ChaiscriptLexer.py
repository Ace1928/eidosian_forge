import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class ChaiscriptLexer(RegexLexer):
    """
    For `ChaiScript <http://chaiscript.com/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'ChaiScript'
    aliases = ['chai', 'chaiscript']
    filenames = ['*.chai']
    mimetypes = ['text/x-chaiscript', 'application/x-chaiscript']
    flags = re.DOTALL | re.MULTILINE
    tokens = {'commentsandwhitespace': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('^\\#.*?\\n', Comment.Single)], 'slashstartsregex': [include('commentsandwhitespace'), ('/(\\\\.|[^[/\\\\\\n]|\\[(\\\\.|[^\\]\\\\\\n])*])+/([gim]+\\b|\\B)', String.Regex, '#pop'), ('(?=/)', Text, ('#pop', 'badregex')), default('#pop')], 'badregex': [('\\n', Text, '#pop')], 'root': [include('commentsandwhitespace'), ('\\n', Text), ('[^\\S\\n]+', Text), ('\\+\\+|--|~|&&|\\?|:|\\|\\||\\\\(?=\\n)|\\.\\.(<<|>>>?|==?|!=?|[-<>+*%&|^/])=?', Operator, 'slashstartsregex'), ('[{(\\[;,]', Punctuation, 'slashstartsregex'), ('[})\\].]', Punctuation), ('[=+\\-*/]', Operator), ('(for|in|while|do|break|return|continue|if|else|throw|try|catch)\\b', Keyword, 'slashstartsregex'), ('(var)\\b', Keyword.Declaration, 'slashstartsregex'), ('(attr|def|fun)\\b', Keyword.Reserved), ('(true|false)\\b', Keyword.Constant), ('(eval|throw)\\b', Name.Builtin), ('`\\S+`', Name.Builtin), ('[$a-zA-Z_]\\w*', Name.Other), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('0x[0-9a-fA-F]+', Number.Hex), ('[0-9]+', Number.Integer), ('"', String.Double, 'dqstring'), ("'(\\\\\\\\|\\\\'|[^'])*'", String.Single)], 'dqstring': [('\\$\\{[^"}]+?\\}', String.Interpol), ('\\$', String.Double), ('\\\\\\\\', String.Double), ('\\\\"', String.Double), ('[^\\\\"$]+', String.Double), ('"', String.Double, '#pop')]}