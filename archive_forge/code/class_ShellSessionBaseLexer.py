import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class ShellSessionBaseLexer(Lexer):
    """
    Base lexer for simplistic shell sessions.

    .. versionadded:: 2.1
    """

    def get_tokens_unprocessed(self, text):
        innerlexer = self._innerLexerCls(**self.options)
        pos = 0
        curcode = ''
        insertions = []
        backslash_continuation = False
        for match in line_re.finditer(text):
            line = match.group()
            m = re.match(self._ps1rgx, line)
            if backslash_continuation:
                curcode += line
                backslash_continuation = curcode.endswith('\\\n')
            elif m:
                if not insertions:
                    pos = match.start()
                insertions.append((len(curcode), [(0, Generic.Prompt, m.group(1))]))
                curcode += m.group(2)
                backslash_continuation = curcode.endswith('\\\n')
            elif line.startswith(self._ps2):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:len(self._ps2)])]))
                curcode += line[len(self._ps2):]
                backslash_continuation = curcode.endswith('\\\n')
            else:
                if insertions:
                    toks = innerlexer.get_tokens_unprocessed(curcode)
                    for i, t, v in do_insertions(insertions, toks):
                        yield (pos + i, t, v)
                yield (match.start(), Generic.Output, line)
                insertions = []
                curcode = ''
        if insertions:
            for i, t, v in do_insertions(insertions, innerlexer.get_tokens_unprocessed(curcode)):
                yield (pos + i, t, v)