import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class DylanConsoleLexer(Lexer):
    """
    For Dylan interactive console output like:

    .. sourcecode:: dylan-console

        ? let a = 1;
        => 1
        ? a
        => 1

    This is based on a copy of the RubyConsoleLexer.

    .. versionadded:: 1.6
    """
    name = 'Dylan session'
    aliases = ['dylan-console', 'dylan-repl']
    filenames = ['*.dylan-console']
    mimetypes = ['text/x-dylan-console']
    _line_re = re.compile('.*?\n')
    _prompt_re = re.compile('\\?| ')

    def get_tokens_unprocessed(self, text):
        dylexer = DylanLexer(**self.options)
        curcode = ''
        insertions = []
        for match in self._line_re.finditer(text):
            line = match.group()
            m = self._prompt_re.match(line)
            if m is not None:
                end = m.end()
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:end])]))
                curcode += line[end:]
            else:
                if curcode:
                    for item in do_insertions(insertions, dylexer.get_tokens_unprocessed(curcode)):
                        yield item
                    curcode = ''
                    insertions = []
                yield (match.start(), Generic.Output, line)
        if curcode:
            for item in do_insertions(insertions, dylexer.get_tokens_unprocessed(curcode)):
                yield item