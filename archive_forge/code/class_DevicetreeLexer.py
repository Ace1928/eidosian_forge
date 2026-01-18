from pygments.lexer import RegexLexer, bygroups, include, default, words
from pygments.token import Comment, Keyword, Name, Number, Operator, \
class DevicetreeLexer(RegexLexer):
    """
    Lexer for Devicetree files.

    .. versionadded:: 2.7
    """
    name = 'Devicetree'
    url = 'https://www.devicetree.org/'
    aliases = ['devicetree', 'dts']
    filenames = ['*.dts', '*.dtsi']
    mimetypes = ['text/x-c']
    _ws = '\\s*(?:/[*][^*/]*?[*]/\\s*)*'
    tokens = {'macro': [('(#include)(' + _ws + ')([^\\n]+)', bygroups(Comment.Preproc, Comment.Multiline, Comment.PreprocFile)), ('(#define)(' + _ws + ')([^\\n]+)', bygroups(Comment.Preproc, Comment.Multiline, Comment.Preproc)), ('(/[^*/{]+/)(' + _ws + ')("[^\\n{]+")', bygroups(Comment.Preproc, Comment.Multiline, Comment.PreprocFile)), ('(/[^*/{]+/)(' + _ws + ')([^\\n;{]*)([;]?)', bygroups(Comment.Preproc, Comment.Multiline, Comment.Preproc, Punctuation))], 'whitespace': [('\\n', Whitespace), ('\\s+', Whitespace), ('\\\\\\n', Text), ('//(\\n|[\\w\\W]*?[^\\\\]\\n)', Comment.Single), ('/(\\\\\\n)?[*][\\w\\W]*?[*](\\\\\\n)?/', Comment.Multiline), ('/(\\\\\\n)?[*][\\w\\W]*', Comment.Multiline)], 'statements': [('(L?)(")', bygroups(String.Affix, String), 'string'), ('0x[0-9a-fA-F]+', Number.Hex), ('\\d+', Number.Integer), ('([^\\s{}/*]*)(\\s*)(:)', bygroups(Name.Label, Text, Punctuation), '#pop'), (words(('compatible', 'model', 'phandle', 'status', '#address-cells', '#size-cells', 'reg', 'virtual-reg', 'ranges', 'dma-ranges', 'device_type', 'name'), suffix='\\b'), Keyword.Reserved), ('([~!%^&*+=|?:<>/#-])', Operator), ('[()\\[\\]{},.]', Punctuation), ('[a-zA-Z_][\\w-]*(?=(?:\\s*,\\s*[a-zA-Z_][\\w-]*|(?:' + _ws + '))*\\s*[=;])', Name), ('[a-zA-Z_]\\w*', Name.Attribute)], 'root': [include('whitespace'), include('macro'), ('([^/*@\\s&]+|/)(@?)((?:0x)?[0-9a-fA-F,]*)(' + _ws + ')(\\{)', bygroups(Name.Function, Operator, Number.Integer, Comment.Multiline, Punctuation), 'node'), default('statement')], 'statement': [include('whitespace'), include('statements'), (';', Punctuation, '#pop')], 'node': [include('whitespace'), include('macro'), ('([^/*@\\s&]+|/)(@?)((?:0x)?[0-9a-fA-F,]*)(' + _ws + ')(\\{)', bygroups(Name.Function, Operator, Number.Integer, Comment.Multiline, Punctuation), '#push'), include('statements'), ('\\};', Punctuation, '#pop'), (';', Punctuation)], 'string': [('"', String, '#pop'), ('\\\\([\\\\abfnrtv"\\\']|x[a-fA-F0-9]{2,4}|u[a-fA-F0-9]{4}|U[a-fA-F0-9]{8}|[0-7]{1,3})', String.Escape), ('[^\\\\"\\n]+', String), ('\\\\\\n', String), ('\\\\', String)]}