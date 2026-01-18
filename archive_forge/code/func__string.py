import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
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