import ast
from pygments.lexer import Lexer, line_re
from pygments.token import Token, Error, Text, Generic
from pygments.util import get_choice_opt
class OutputLexer(Lexer):
    """
    Simple lexer that highlights everything as ``Token.Generic.Output``.

    .. versionadded:: 2.10
    """
    name = 'Text output'
    aliases = ['output']

    def get_tokens_unprocessed(self, text):
        yield (0, Generic.Output, text)