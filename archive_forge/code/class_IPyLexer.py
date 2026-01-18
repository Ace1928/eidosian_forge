import re
from pygments.lexers import (
from pygments.lexer import (
from pygments.token import (
from pygments.util import get_bool_opt
class IPyLexer(Lexer):
    """
    Primary lexer for all IPython-like code.

    This is a simple helper lexer.  If the first line of the text begins with
    "In \\[[0-9]+\\]:", then the entire text is parsed with an IPython console
    lexer. If not, then the entire text is parsed with an IPython lexer.

    The goal is to reduce the number of lexers that are registered
    with Pygments.

    """
    name = 'IPy session'
    aliases = ['ipy']

    def __init__(self, **options):
        """
        Create a new IPyLexer instance which dispatch to either an
        IPythonCOnsoleLexer (if In prompts are present) or and IPythonLexer (if
        In prompts are not present).
        """
        self.python3 = get_bool_opt(options, 'python3', False)
        if self.python3:
            self.aliases = ['ipy3']
        else:
            self.aliases = ['ipy2', 'ipy']
        Lexer.__init__(self, **options)
        self.IPythonLexer = IPythonLexer(**options)
        self.IPythonConsoleLexer = IPythonConsoleLexer(**options)

    def get_tokens_unprocessed(self, text):
        if re.match('.*(In \\[[0-9]+\\]:)', text.strip(), re.DOTALL):
            lex = self.IPythonConsoleLexer
        else:
            lex = self.IPythonLexer
        for token in lex.get_tokens_unprocessed(text):
            yield token