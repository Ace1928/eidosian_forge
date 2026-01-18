import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, words
from pygments.token import Punctuation, Whitespace, Error, \
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pygments.util import iteritems
from pygments.lexers._postgres_builtins import KEYWORDS, DATATYPES, \
from pygments.lexers import _tsql_builtins
class PostgresConsoleLexer(Lexer):
    """
    Lexer for psql sessions.

    .. versionadded:: 1.5
    """
    name = 'PostgreSQL console (psql)'
    aliases = ['psql', 'postgresql-console', 'postgres-console']
    mimetypes = ['text/x-postgresql-psql']

    def get_tokens_unprocessed(self, data):
        sql = PsqlRegexLexer(**self.options)
        lines = lookahead(line_re.findall(data))
        while 1:
            curcode = ''
            insertions = []
            while 1:
                try:
                    line = next(lines)
                except StopIteration:
                    break
                if line.startswith('$') and (not curcode):
                    lexer = get_lexer_by_name('console', **self.options)
                    for x in lexer.get_tokens_unprocessed(line):
                        yield x
                    break
                mprompt = re_prompt.match(line)
                if mprompt is not None:
                    insertions.append((len(curcode), [(0, Generic.Prompt, mprompt.group())]))
                    curcode += line[len(mprompt.group()):]
                else:
                    curcode += line
                if re_psql_command.match(curcode) or re_end_command.search(curcode):
                    break
            for item in do_insertions(insertions, sql.get_tokens_unprocessed(curcode)):
                yield item
            out_token = Generic.Output
            while 1:
                line = next(lines)
                mprompt = re_prompt.match(line)
                if mprompt is not None:
                    lines.send(line)
                    break
                mmsg = re_message.match(line)
                if mmsg is not None:
                    if mmsg.group(1).startswith('ERROR') or mmsg.group(1).startswith('FATAL'):
                        out_token = Generic.Error
                    yield (mmsg.start(1), Generic.Strong, mmsg.group(1))
                    yield (mmsg.start(2), out_token, mmsg.group(2))
                else:
                    yield (0, out_token, line)