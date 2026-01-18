import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class KconfigLexer(RegexLexer):
    """
    For Linux-style Kconfig files.

    .. versionadded:: 1.6
    """
    name = 'Kconfig'
    aliases = ['kconfig', 'menuconfig', 'linux-config', 'kernel-config']
    filenames = ['Kconfig', '*Config.in*', 'external.in*', 'standard-modules.in']
    mimetypes = ['text/x-kconfig']
    flags = 0

    def call_indent(level):
        return (_rx_indent(level), String.Doc, 'indent%s' % level)

    def do_indent(level):
        return [(_rx_indent(level), String.Doc), ('\\s*\\n', Text), default('#pop:2')]
    tokens = {'root': [('\\s+', Text), ('#.*?\\n', Comment.Single), (words(('mainmenu', 'config', 'menuconfig', 'choice', 'endchoice', 'comment', 'menu', 'endmenu', 'visible if', 'if', 'endif', 'source', 'prompt', 'select', 'depends on', 'default', 'range', 'option'), suffix='\\b'), Keyword), ('(---help---|help)[\\t ]*\\n', Keyword, 'help'), ('(bool|tristate|string|hex|int|defconfig_list|modules|env)\\b', Name.Builtin), ('[!=&|]', Operator), ('[()]', Punctuation), ('[0-9]+', Number.Integer), ("'(''|[^'])*'", String.Single), ('"(""|[^"])*"', String.Double), ('\\S+', Text)], 'help': [('\\s*\\n', Text), call_indent(7), call_indent(6), call_indent(5), call_indent(4), call_indent(3), call_indent(2), call_indent(1), default('#pop')], 'indent7': do_indent(7), 'indent6': do_indent(6), 'indent5': do_indent(5), 'indent4': do_indent(4), 'indent3': do_indent(3), 'indent2': do_indent(2), 'indent1': do_indent(1)}