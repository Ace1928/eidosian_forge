import re
from pygments.lexer import Lexer
from pygments.util import get_bool_opt, get_list_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.scanner import Scanner
from pygments.lexers.modula2 import Modula2Lexer
class PortugolLexer(Lexer):
    """For Portugol, a Pascal dialect with keywords in Portuguese."""
    name = 'Portugol'
    aliases = ['portugol']
    filenames = ['*.alg', '*.portugol']
    mimetypes = []
    url = 'https://www.apoioinformatica.inf.br/produtos/visualg/linguagem'

    def __init__(self, **options):
        Lexer.__init__(self, **options)
        self.lexer = DelphiLexer(**options, portugol=True)

    def get_tokens_unprocessed(self, text):
        return self.lexer.get_tokens_unprocessed(text)