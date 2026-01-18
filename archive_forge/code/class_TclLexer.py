from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
class TclLexer(RegexLexer):
    """
    For Tcl source code.

    .. versionadded:: 0.10
    """
    keyword_cmds_re = words(('after', 'apply', 'array', 'break', 'catch', 'continue', 'elseif', 'else', 'error', 'eval', 'expr', 'for', 'foreach', 'global', 'if', 'namespace', 'proc', 'rename', 'return', 'set', 'switch', 'then', 'trace', 'unset', 'update', 'uplevel', 'upvar', 'variable', 'vwait', 'while'), prefix='\\b', suffix='\\b')
    builtin_cmds_re = words(('append', 'bgerror', 'binary', 'cd', 'chan', 'clock', 'close', 'concat', 'dde', 'dict', 'encoding', 'eof', 'exec', 'exit', 'fblocked', 'fconfigure', 'fcopy', 'file', 'fileevent', 'flush', 'format', 'gets', 'glob', 'history', 'http', 'incr', 'info', 'interp', 'join', 'lappend', 'lassign', 'lindex', 'linsert', 'list', 'llength', 'load', 'loadTk', 'lrange', 'lrepeat', 'lreplace', 'lreverse', 'lsearch', 'lset', 'lsort', 'mathfunc', 'mathop', 'memory', 'msgcat', 'open', 'package', 'pid', 'pkg::create', 'pkg_mkIndex', 'platform', 'platform::shell', 'puts', 'pwd', 're_syntax', 'read', 'refchan', 'regexp', 'registry', 'regsub', 'scan', 'seek', 'socket', 'source', 'split', 'string', 'subst', 'tell', 'time', 'tm', 'unknown', 'unload'), prefix='\\b', suffix='\\b')
    name = 'Tcl'
    aliases = ['tcl']
    filenames = ['*.tcl', '*.rvt']
    mimetypes = ['text/x-tcl', 'text/x-script.tcl', 'application/x-tcl']

    def _gen_command_rules(keyword_cmds_re, builtin_cmds_re, context=''):
        return [(keyword_cmds_re, Keyword, 'params' + context), (builtin_cmds_re, Name.Builtin, 'params' + context), ('([\\w.-]+)', Name.Variable, 'params' + context), ('#', Comment, 'comment')]
    tokens = {'root': [include('command'), include('basic'), include('data'), ('\\}', Keyword)], 'command': _gen_command_rules(keyword_cmds_re, builtin_cmds_re), 'command-in-brace': _gen_command_rules(keyword_cmds_re, builtin_cmds_re, '-in-brace'), 'command-in-bracket': _gen_command_rules(keyword_cmds_re, builtin_cmds_re, '-in-bracket'), 'command-in-paren': _gen_command_rules(keyword_cmds_re, builtin_cmds_re, '-in-paren'), 'basic': [('\\(', Keyword, 'paren'), ('\\[', Keyword, 'bracket'), ('\\{', Keyword, 'brace'), ('"', String.Double, 'string'), ('(eq|ne|in|ni)\\b', Operator.Word), ('!=|==|<<|>>|<=|>=|&&|\\|\\||\\*\\*|[-+~!*/%<>&^|?:]', Operator)], 'data': [('\\s+', Text), ('0x[a-fA-F0-9]+', Number.Hex), ('0[0-7]+', Number.Oct), ('\\d+\\.\\d+', Number.Float), ('\\d+', Number.Integer), ('\\$([\\w.:-]+)', Name.Variable), ('([\\w.:-]+)', Text)], 'params': [(';', Keyword, '#pop'), ('\\n', Text, '#pop'), ('(else|elseif|then)\\b', Keyword), include('basic'), include('data')], 'params-in-brace': [('\\}', Keyword, ('#pop', '#pop')), include('params')], 'params-in-paren': [('\\)', Keyword, ('#pop', '#pop')), include('params')], 'params-in-bracket': [('\\]', Keyword, ('#pop', '#pop')), include('params')], 'string': [('\\[', String.Double, 'string-square'), ('(?s)(\\\\\\\\|\\\\[0-7]+|\\\\.|[^"\\\\])', String.Double), ('"', String.Double, '#pop')], 'string-square': [('\\[', String.Double, 'string-square'), ('(?s)(\\\\\\\\|\\\\[0-7]+|\\\\.|\\\\\\n|[^\\]\\\\])', String.Double), ('\\]', String.Double, '#pop')], 'brace': [('\\}', Keyword, '#pop'), include('command-in-brace'), include('basic'), include('data')], 'paren': [('\\)', Keyword, '#pop'), include('command-in-paren'), include('basic'), include('data')], 'bracket': [('\\]', Keyword, '#pop'), include('command-in-bracket'), include('basic'), include('data')], 'comment': [('.*[^\\\\]\\n', Comment, '#pop'), ('.*\\\\\\n', Comment)]}

    def analyse_text(text):
        return shebang_matches(text, '(tcl)')