import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class NestedTextLexer(RegexLexer):
    """
    Lexer for *NextedText*, a human-friendly data format.

    .. versionadded:: 2.9

    .. versionchanged:: 2.16
        Added support for *NextedText* v3.0.
    """
    name = 'NestedText'
    url = 'https://nestedtext.org'
    aliases = ['nestedtext', 'nt']
    filenames = ['*.nt']
    tokens = {'root': [('^([ ]*)(#.*)$', bygroups(Whitespace, Comment)), ('^([ ]*)(\\{)', bygroups(Whitespace, Punctuation), 'inline_dict'), ('^([ ]*)(\\[)', bygroups(Whitespace, Punctuation), 'inline_list'), ('^([ ]*)(>)$', bygroups(Whitespace, Punctuation)), ('^([ ]*)(>)( )(.*?)([ \\t]*)$', bygroups(Whitespace, Punctuation, Whitespace, Text, Whitespace)), ('^([ ]*)(-)$', bygroups(Whitespace, Punctuation)), ('^([ ]*)(-)( )(.*?)([ \\t]*)$', bygroups(Whitespace, Punctuation, Whitespace, Text, Whitespace)), ('^([ ]*)(:)$', bygroups(Whitespace, Punctuation)), ('^([ ]*)(:)( )([^\\n]*?)([ \\t]*)$', bygroups(Whitespace, Punctuation, Whitespace, Name.Tag, Whitespace)), ('^([ ]*)([^\\{\\[\\s].*?)(:)$', bygroups(Whitespace, Name.Tag, Punctuation)), ('^([ ]*)([^\\{\\[\\s].*?)(:)( )(.*?)([ \\t]*)$', bygroups(Whitespace, Name.Tag, Punctuation, Whitespace, Text, Whitespace))], 'inline_list': [include('whitespace'), ('[^\\{\\}\\[\\],\\s]+', Text), include('inline_value'), (',', Punctuation), ('\\]', Punctuation, '#pop'), ('\\n', Error, '#pop')], 'inline_dict': [include('whitespace'), ('[^\\{\\}\\[\\],:\\s]+', Name.Tag), (':', Punctuation, 'inline_dict_value'), ('\\}', Punctuation, '#pop'), ('\\n', Error, '#pop')], 'inline_dict_value': [include('whitespace'), ('[^\\{\\}\\[\\],:\\s]+', Text), include('inline_value'), (',', Punctuation, '#pop'), ('\\}', Punctuation, '#pop:2')], 'inline_value': [include('whitespace'), ('\\{', Punctuation, 'inline_dict'), ('\\[', Punctuation, 'inline_list')], 'whitespace': [('[ \\t]+', Whitespace)]}