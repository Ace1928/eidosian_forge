import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class SnowballLexer(ExtendedRegexLexer):
    """
    Lexer for `Snowball <http://snowballstem.org/>`_ source code.

    .. versionadded:: 2.2
    """
    name = 'Snowball'
    aliases = ['snowball']
    filenames = ['*.sbl']
    _ws = '\\n\\r\\t '

    def __init__(self, **options):
        self._reset_stringescapes()
        ExtendedRegexLexer.__init__(self, **options)

    def _reset_stringescapes(self):
        self._start = "'"
        self._end = "'"

    def _string(do_string_first):

        def callback(lexer, match, ctx):
            s = match.start()
            text = match.group()
            string = re.compile('([^%s]*)(.)' % re.escape(lexer._start)).match
            escape = re.compile('([^%s]*)(.)' % re.escape(lexer._end)).match
            pos = 0
            do_string = do_string_first
            while pos < len(text):
                if do_string:
                    match = string(text, pos)
                    yield (s + match.start(1), String.Single, match.group(1))
                    if match.group(2) == "'":
                        yield (s + match.start(2), String.Single, match.group(2))
                        ctx.stack.pop()
                        break
                    yield (s + match.start(2), String.Escape, match.group(2))
                    pos = match.end()
                match = escape(text, pos)
                yield (s + match.start(), String.Escape, match.group())
                if match.group(2) != lexer._end:
                    ctx.stack[-1] = 'escape'
                    break
                pos = match.end()
                do_string = True
            ctx.pos = s + match.end()
        return callback

    def _stringescapes(lexer, match, ctx):
        lexer._start = match.group(3)
        lexer._end = match.group(5)
        return bygroups(Keyword.Reserved, Text, String.Escape, Text, String.Escape)(lexer, match, ctx)
    tokens = {'root': [(words(('len', 'lenof'), suffix='\\b'), Operator.Word), include('root1')], 'root1': [('[%s]+' % _ws, Text), ('\\d+', Number.Integer), ("'", String.Single, 'string'), ('[()]', Punctuation), ('/\\*[\\w\\W]*?\\*/', Comment.Multiline), ('//.*', Comment.Single), ('[!*+\\-/<=>]=|[-=]>|<[+-]|[$*+\\-/<=>?\\[\\]]', Operator), (words(('as', 'get', 'hex', 'among', 'define', 'decimal', 'backwardmode'), suffix='\\b'), Keyword.Reserved), (words(('strings', 'booleans', 'integers', 'routines', 'externals', 'groupings'), suffix='\\b'), Keyword.Reserved, 'declaration'), (words(('do', 'or', 'and', 'for', 'hop', 'non', 'not', 'set', 'try', 'fail', 'goto', 'loop', 'next', 'test', 'true', 'false', 'unset', 'atmark', 'attach', 'delete', 'gopast', 'insert', 'repeat', 'sizeof', 'tomark', 'atleast', 'atlimit', 'reverse', 'setmark', 'tolimit', 'setlimit', 'backwards', 'substring'), suffix='\\b'), Operator.Word), (words(('size', 'limit', 'cursor', 'maxint', 'minint'), suffix='\\b'), Name.Builtin), ('(stringdef\\b)([%s]*)([^%s]+)' % (_ws, _ws), bygroups(Keyword.Reserved, Text, String.Escape)), ('(stringescapes\\b)([%s]*)(.)([%s]*)(.)' % (_ws, _ws), _stringescapes), ('[A-Za-z]\\w*', Name)], 'declaration': [('\\)', Punctuation, '#pop'), (words(('len', 'lenof'), suffix='\\b'), Name, ('root1', 'declaration')), include('root1')], 'string': [("[^']*'", _string(True))], 'escape': [("[^']*'", _string(False))]}

    def get_tokens_unprocessed(self, text=None, context=None):
        self._reset_stringescapes()
        return ExtendedRegexLexer.get_tokens_unprocessed(self, text, context)