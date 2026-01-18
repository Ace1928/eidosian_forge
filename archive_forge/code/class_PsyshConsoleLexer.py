import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, default, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, get_list_opt, shebang_matches
class PsyshConsoleLexer(Lexer):
    """
    For PsySH console output, such as:

    .. sourcecode:: psysh

        >>> $greeting = function($name): string {
        ...     return "Hello, {$name}";
        ... };
        => Closure($name): string {#2371 …3}
        >>> $greeting('World')
        => "Hello, World"

    .. versionadded:: 2.7
    """
    name = 'PsySH console session for PHP'
    url = 'https://psysh.org/'
    aliases = ['psysh']

    def __init__(self, **options):
        options['startinline'] = True
        Lexer.__init__(self, **options)

    def get_tokens_unprocessed(self, text):
        phplexer = PhpLexer(**self.options)
        curcode = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            if line.startswith('>>> ') or line.startswith('... '):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:4])]))
                curcode += line[4:]
            elif line.rstrip() == '...':
                insertions.append((len(curcode), [(0, Generic.Prompt, '...')]))
                curcode += line[3:]
            else:
                if curcode:
                    yield from do_insertions(insertions, phplexer.get_tokens_unprocessed(curcode))
                    curcode = ''
                    insertions = []
                yield (match.start(), Generic.Output, line)
        if curcode:
            yield from do_insertions(insertions, phplexer.get_tokens_unprocessed(curcode))