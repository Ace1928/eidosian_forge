import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _is_closing(self, current_token, open_token):
    return (current_token.type == TOKEN.END_BLOCK or current_token.type == TOKEN.END_EXPR) and (open_token is not None and (current_token.text == ']' and open_token.text == '[' or (current_token.text == ')' and open_token.text == '(') or (current_token.text == '}' and open_token.text == '{')))