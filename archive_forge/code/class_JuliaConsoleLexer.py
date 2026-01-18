import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches, unirange
class JuliaConsoleLexer(Lexer):
    """
    For Julia console sessions. Modeled after MatlabSessionLexer.

    .. versionadded:: 1.6
    """
    name = 'Julia console'
    aliases = ['jlcon']

    def get_tokens_unprocessed(self, text):
        jllexer = JuliaLexer(**self.options)
        start = 0
        curcode = ''
        insertions = []
        output = False
        error = False
        for line in text.splitlines(True):
            if line.startswith('julia>'):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:6])]))
                curcode += line[6:]
                output = False
                error = False
            elif line.startswith('help?>') or line.startswith('shell>'):
                yield (start, Generic.Prompt, line[:6])
                yield (start + 6, Text, line[6:])
                output = False
                error = False
            elif line.startswith('      ') and (not output):
                insertions.append((len(curcode), [(0, Text, line[:6])]))
                curcode += line[6:]
            else:
                if curcode:
                    for item in do_insertions(insertions, jllexer.get_tokens_unprocessed(curcode)):
                        yield item
                    curcode = ''
                    insertions = []
                if line.startswith('ERROR: ') or error:
                    yield (start, Generic.Error, line)
                    error = True
                else:
                    yield (start, Generic.Output, line)
                output = True
            start += len(line)
        if curcode:
            for item in do_insertions(insertions, jllexer.get_tokens_unprocessed(curcode)):
                yield item