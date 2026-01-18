import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class PanLexer(RegexLexer):
    """
    Lexer for `pan <http://github.com/quattor/pan/>`_ source files.

    Based on tcsh lexer.

    .. versionadded:: 2.0
    """
    name = 'Pan'
    aliases = ['pan']
    filenames = ['*.pan']
    tokens = {'root': [include('basic'), ('\\(', Keyword, 'paren'), ('\\{', Keyword, 'curly'), include('data')], 'basic': [(words(('if', 'for', 'with', 'else', 'type', 'bind', 'while', 'valid', 'final', 'prefix', 'unique', 'object', 'foreach', 'include', 'template', 'function', 'variable', 'structure', 'extensible', 'declaration'), prefix='\\b', suffix='\\s*\\b'), Keyword), (words(('file_contents', 'format', 'index', 'length', 'match', 'matches', 'replace', 'splice', 'split', 'substr', 'to_lowercase', 'to_uppercase', 'debug', 'error', 'traceback', 'deprecated', 'base64_decode', 'base64_encode', 'digest', 'escape', 'unescape', 'append', 'create', 'first', 'nlist', 'key', 'list', 'merge', 'next', 'prepend', 'is_boolean', 'is_defined', 'is_double', 'is_list', 'is_long', 'is_nlist', 'is_null', 'is_number', 'is_property', 'is_resource', 'is_string', 'to_boolean', 'to_double', 'to_long', 'to_string', 'clone', 'delete', 'exists', 'path_exists', 'if_exists', 'return', 'value'), prefix='\\b', suffix='\\s*\\b'), Name.Builtin), ('#.*', Comment), ('\\\\[\\w\\W]', String.Escape), ('(\\b\\w+)(\\s*)(=)', bygroups(Name.Variable, Text, Operator)), ('[\\[\\]{}()=]+', Operator), ("<<\\s*(\\'?)\\\\?(\\w+)[\\w\\W]+?\\2", String), (';', Punctuation)], 'data': [('(?s)"(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\])*"', String.Double), ("(?s)'(\\\\\\\\|\\\\[0-7]+|\\\\.|[^'\\\\])*'", String.Single), ('\\s+', Text), ('[^=\\s\\[\\]{}()$"\\\'`\\\\;#]+', Text), ('\\d+(?= |\\Z)', Number)], 'curly': [('\\}', Keyword, '#pop'), (':-', Keyword), ('\\w+', Name.Variable), ('[^}:"\\\'`$]+', Punctuation), (':', Punctuation), include('root')], 'paren': [('\\)', Keyword, '#pop'), include('root')]}