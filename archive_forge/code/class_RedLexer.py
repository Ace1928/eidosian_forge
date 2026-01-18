import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RedLexer(RegexLexer):
    """
    A `Red-language <http://www.red-lang.org/>`_ lexer.

    .. versionadded:: 2.0
    """
    name = 'Red'
    aliases = ['red', 'red/system']
    filenames = ['*.red', '*.reds']
    mimetypes = ['text/x-red', 'text/x-red-system']
    flags = re.IGNORECASE | re.MULTILINE
    escape_re = '(?:\\^\\([0-9a-f]{1,4}\\)*)'

    def word_callback(lexer, match):
        word = match.group()
        if re.match('.*:$', word):
            yield (match.start(), Generic.Subheading, word)
        elif re.match('(if|unless|either|any|all|while|until|loop|repeat|foreach|forall|func|function|does|has|switch|case|reduce|compose|get|set|print|prin|equal\\?|not-equal\\?|strict-equal\\?|lesser\\?|greater\\?|lesser-or-equal\\?|greater-or-equal\\?|same\\?|not|type\\?|stats|bind|union|replace|charset|routine)$', word):
            yield (match.start(), Name.Builtin, word)
        elif re.match('(make|random|reflect|to|form|mold|absolute|add|divide|multiply|negate|power|remainder|round|subtract|even\\?|odd\\?|and~|complement|or~|xor~|append|at|back|change|clear|copy|find|head|head\\?|index\\?|insert|length\\?|next|pick|poke|remove|reverse|select|sort|skip|swap|tail|tail\\?|take|trim|create|close|delete|modify|open|open\\?|query|read|rename|update|write)$', word):
            yield (match.start(), Name.Function, word)
        elif re.match('(yes|on|no|off|true|false|tab|cr|lf|newline|escape|slash|sp|space|null|none|crlf|dot|null-byte)$', word):
            yield (match.start(), Name.Builtin.Pseudo, word)
        elif re.match('(#system-global|#include|#enum|#define|#either|#if|#import|#export|#switch|#default|#get-definition)$', word):
            yield (match.start(), Keyword.Namespace, word)
        elif re.match('(system|halt|quit|quit-return|do|load|q|recycle|call|run|ask|parse|raise-error|return|exit|break|alias|push|pop|probe|\\?\\?|spec-of|body-of|quote|forever)$', word):
            yield (match.start(), Name.Exception, word)
        elif re.match('(action\\?|block\\?|char\\?|datatype\\?|file\\?|function\\?|get-path\\?|zero\\?|get-word\\?|integer\\?|issue\\?|lit-path\\?|lit-word\\?|logic\\?|native\\?|op\\?|paren\\?|path\\?|refinement\\?|set-path\\?|set-word\\?|string\\?|unset\\?|any-struct\\?|none\\?|word\\?|any-series\\?)$', word):
            yield (match.start(), Keyword, word)
        elif re.match('(JNICALL|stdcall|cdecl|infix)$', word):
            yield (match.start(), Keyword.Namespace, word)
        elif re.match('to-.*', word):
            yield (match.start(), Keyword, word)
        elif re.match('(\\+|-\\*\\*|-|\\*\\*|//|/|\\*|and|or|xor|=\\?|===|==|=|<>|<=|>=|<<<|>>>|<<|>>|<|>%)$', word):
            yield (match.start(), Operator, word)
        elif re.match('.*\\!$', word):
            yield (match.start(), Keyword.Type, word)
        elif re.match("'.*", word):
            yield (match.start(), Name.Variable.Instance, word)
        elif re.match('#.*', word):
            yield (match.start(), Name.Label, word)
        elif re.match('%.*', word):
            yield (match.start(), Name.Decorator, word)
        elif re.match(':.*', word):
            yield (match.start(), Generic.Subheading, word)
        else:
            yield (match.start(), Name.Variable, word)
    tokens = {'root': [('[^R]+', Comment), ('Red/System\\s+\\[', Generic.Strong, 'script'), ('Red\\s+\\[', Generic.Strong, 'script'), ('R', Comment)], 'script': [('\\s+', Text), ('#"', String.Char, 'char'), ('#\\{[0-9a-f\\s]*\\}', Number.Hex), ('2#\\{', Number.Hex, 'bin2'), ('64#\\{[0-9a-z+/=\\s]*\\}', Number.Hex), ('([0-9a-f]+)(h)((\\s)|(?=[\\[\\]{}"()]))', bygroups(Number.Hex, Name.Variable, Whitespace)), ('"', String, 'string'), ('\\{', String, 'string2'), (';#+.*\\n', Comment.Special), (';\\*+.*\\n', Comment.Preproc), (';.*\\n', Comment), ('%"', Name.Decorator, 'stringFile'), ('%[^(^{")\\s\\[\\]]+', Name.Decorator), ('[+-]?([a-z]{1,3})?\\$\\d+(\\.\\d+)?', Number.Float), ('[+-]?\\d+\\:\\d+(\\:\\d+)?(\\.\\d+)?', String.Other), ('\\d+[\\-/][0-9a-z]+[\\-/]\\d+(/\\d+:\\d+((:\\d+)?([\\.\\d+]?([+-]?\\d+:\\d+)?)?)?)?', String.Other), ('\\d+(\\.\\d+)+\\.\\d+', Keyword.Constant), ('\\d+X\\d+', Keyword.Constant), ("[+-]?\\d+(\\'\\d+)?([.,]\\d*)?E[+-]?\\d+", Number.Float), ("[+-]?\\d+(\\'\\d+)?[.,]\\d*", Number.Float), ("[+-]?\\d+(\\'\\d+)?", Number), ('[\\[\\]()]', Generic.Strong), ('[a-z]+[^(^{"\\s:)]*://[^(^{"\\s)]*', Name.Decorator), ('mailto:[^(^{"@\\s)]+@[^(^{"@\\s)]+', Name.Decorator), ('[^(^{"@\\s)]+@[^(^{"@\\s)]+', Name.Decorator), ('comment\\s"', Comment, 'commentString1'), ('comment\\s\\{', Comment, 'commentString2'), ('comment\\s\\[', Comment, 'commentBlock'), ('comment\\s[^(\\s{"\\[]+', Comment), ('/[^(^{^")\\s/[\\]]*', Name.Attribute), ('([^(^{^")\\s/[\\]]+)(?=[:({"\\s/\\[\\]])', word_callback), ('<[\\w:.-]*>', Name.Tag), ('<[^(<>\\s")]+', Name.Tag, 'tag'), ('([^(^{")\\s]+)', Text)], 'string': [('[^(^")]+', String), (escape_re, String.Escape), ('[(|)]+', String), ('\\^.', String.Escape), ('"', String, '#pop')], 'string2': [('[^(^{})]+', String), (escape_re, String.Escape), ('[(|)]+', String), ('\\^.', String.Escape), ('\\{', String, '#push'), ('\\}', String, '#pop')], 'stringFile': [('[^(^")]+', Name.Decorator), (escape_re, Name.Decorator), ('\\^.', Name.Decorator), ('"', Name.Decorator, '#pop')], 'char': [(escape_re + '"', String.Char, '#pop'), ('\\^."', String.Char, '#pop'), ('."', String.Char, '#pop')], 'tag': [(escape_re, Name.Tag), ('"', Name.Tag, 'tagString'), ('[^(<>\\r\\n")]+', Name.Tag), ('>', Name.Tag, '#pop')], 'tagString': [('[^(^")]+', Name.Tag), (escape_re, Name.Tag), ('[(|)]+', Name.Tag), ('\\^.', Name.Tag), ('"', Name.Tag, '#pop')], 'tuple': [('(\\d+\\.)+', Keyword.Constant), ('\\d+', Keyword.Constant, '#pop')], 'bin2': [('\\s+', Number.Hex), ('([01]\\s*){8}', Number.Hex), ('\\}', Number.Hex, '#pop')], 'commentString1': [('[^(^")]+', Comment), (escape_re, Comment), ('[(|)]+', Comment), ('\\^.', Comment), ('"', Comment, '#pop')], 'commentString2': [('[^(^{})]+', Comment), (escape_re, Comment), ('[(|)]+', Comment), ('\\^.', Comment), ('\\{', Comment, '#push'), ('\\}', Comment, '#pop')], 'commentBlock': [('\\[', Comment, '#push'), ('\\]', Comment, '#pop'), ('"', Comment, 'commentString1'), ('\\{', Comment, 'commentString2'), ('[^(\\[\\]"{)]+', Comment)]}