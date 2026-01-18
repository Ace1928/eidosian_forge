from pygments.lexer import RegexLexer, include, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class FloScriptLexer(RegexLexer):
    """
    For FloScript configuration language source code.

    .. versionadded:: 2.4
    """
    name = 'FloScript'
    url = 'https://github.com/ioflo/ioflo'
    aliases = ['floscript', 'flo']
    filenames = ['*.flo']

    def innerstring_rules(ttype):
        return [('%(\\(\\w+\\))?[-#0 +]*([0-9]+|[*])?(\\.([0-9]+|[*]))?[hlL]?[E-GXc-giorsux%]', String.Interpol), ('[^\\\\\\\'"%\\n]+', ttype), ('[\\\'"\\\\]', ttype), ('%', ttype)]
    tokens = {'root': [('\\s+', Whitespace), ('[]{}:(),;[]', Punctuation), ('(\\\\)(\\n)', bygroups(Text, Whitespace)), ('\\\\', Text), ('(to|by|with|from|per|for|cum|qua|via|as|at|in|of|on|re|is|if|be|into|and|not)\\b', Operator.Word), ('!=|==|<<|>>|[-~+/*%=<>&^|.]', Operator), ('(load|init|server|logger|log|loggee|first|over|under|next|done|timeout|repeat|native|benter|enter|recur|exit|precur|renter|rexit|print|put|inc|copy|set|aux|rear|raze|go|let|do|bid|ready|start|stop|run|abort|use|flo|give|take)\\b', Name.Builtin), ('(frame|framer|house)\\b', Keyword), ('"', String, 'string'), include('name'), include('numbers'), ('#.+$', Comment.Single)], 'string': [('[^"]+', String), ('"', String, '#pop')], 'numbers': [('(\\d+\\.\\d*|\\d*\\.\\d+)([eE][+-]?[0-9]+)?j?', Number.Float), ('\\d+[eE][+-]?[0-9]+j?', Number.Float), ('0[0-7]+j?', Number.Oct), ('0[bB][01]+', Number.Bin), ('0[xX][a-fA-F0-9]+', Number.Hex), ('\\d+L', Number.Integer.Long), ('\\d+j?', Number.Integer)], 'name': [('@[\\w.]+', Name.Decorator), ('[a-zA-Z_]\\w*', Name)]}