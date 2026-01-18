import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class ImportSetting(Tokenizer):
    _tokens = (IMPORT, ARGUMENT)