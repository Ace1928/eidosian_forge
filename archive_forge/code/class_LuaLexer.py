import re
from pygments.lexer import RegexLexer, include, bygroups, default, combined, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, iteritems
class LuaLexer(RegexLexer):
    """
    For `Lua <http://www.lua.org>`_ source code.

    Additional options accepted:

    `func_name_highlighting`
        If given and ``True``, highlight builtin function names
        (default: ``True``).
    `disabled_modules`
        If given, must be a list of module names whose function names
        should not be highlighted. By default all modules are highlighted.

        To get a list of allowed modules have a look into the
        `_lua_builtins` module:

        .. sourcecode:: pycon

            >>> from pygments.lexers._lua_builtins import MODULES
            >>> MODULES.keys()
            ['string', 'coroutine', 'modules', 'io', 'basic', ...]
    """
    name = 'Lua'
    aliases = ['lua']
    filenames = ['*.lua', '*.wlua']
    mimetypes = ['text/x-lua', 'application/x-lua']
    _comment_multiline = '(?:--\\[(?P<level>=*)\\[[\\w\\W]*?\\](?P=level)\\])'
    _comment_single = '(?:--.*$)'
    _space = '(?:\\s+)'
    _s = '(?:%s|%s|%s)' % (_comment_multiline, _comment_single, _space)
    _name = '(?:[^\\W\\d]\\w*)'
    tokens = {'root': [('#!.*', Comment.Preproc), default('base')], 'ws': [(_comment_multiline, Comment.Multiline), (_comment_single, Comment.Single), (_space, Text)], 'base': [include('ws'), ('(?i)0x[\\da-f]*(\\.[\\da-f]*)?(p[+-]?\\d+)?', Number.Hex), ('(?i)(\\d*\\.\\d+|\\d+\\.\\d*)(e[+-]?\\d+)?', Number.Float), ('(?i)\\d+e[+-]?\\d+', Number.Float), ('\\d+', Number.Integer), ('(?s)\\[(=*)\\[.*?\\]\\1\\]', String), ('::', Punctuation, 'label'), ('\\.{3}', Punctuation), ('[=<>|~&+\\-*/%#^]+|\\.\\.', Operator), ('[\\[\\]{}().,:;]', Punctuation), ('(and|or|not)\\b', Operator.Word), ('(break|do|else|elseif|end|for|if|in|repeat|return|then|until|while)\\b', Keyword.Reserved), ('goto\\b', Keyword.Reserved, 'goto'), ('(local)\\b', Keyword.Declaration), ('(true|false|nil)\\b', Keyword.Constant), ('(function)\\b', Keyword.Reserved, 'funcname'), ('[A-Za-z_]\\w*(\\.[A-Za-z_]\\w*)?', Name), ("'", String.Single, combined('stringescape', 'sqs')), ('"', String.Double, combined('stringescape', 'dqs'))], 'funcname': [include('ws'), ('[.:]', Punctuation), ('%s(?=%s*[.:])' % (_name, _s), Name.Class), (_name, Name.Function, '#pop'), ('\\(', Punctuation, '#pop')], 'goto': [include('ws'), (_name, Name.Label, '#pop')], 'label': [include('ws'), ('::', Punctuation, '#pop'), (_name, Name.Label)], 'stringescape': [('\\\\([abfnrtv\\\\"\\\']|[\\r\\n]{1,2}|z\\s*|x[0-9a-fA-F]{2}|\\d{1,3}|u\\{[0-9a-fA-F]+\\})', String.Escape)], 'sqs': [("'", String.Single, '#pop'), ("[^\\\\']+", String.Single)], 'dqs': [('"', String.Double, '#pop'), ('[^\\\\"]+', String.Double)]}

    def __init__(self, **options):
        self.func_name_highlighting = get_bool_opt(options, 'func_name_highlighting', True)
        self.disabled_modules = get_list_opt(options, 'disabled_modules', [])
        self._functions = set()
        if self.func_name_highlighting:
            from pygments.lexers._lua_builtins import MODULES
            for mod, func in iteritems(MODULES):
                if mod not in self.disabled_modules:
                    self._functions.update(func)
        RegexLexer.__init__(self, **options)

    def get_tokens_unprocessed(self, text):
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text):
            if token is Name:
                if value in self._functions:
                    yield (index, Name.Builtin, value)
                    continue
                elif '.' in value:
                    a, b = value.split('.')
                    yield (index, Name, a)
                    yield (index + len(a), Punctuation, u'.')
                    yield (index + len(a) + 1, Name, b)
                    continue
            yield (index, token, value)