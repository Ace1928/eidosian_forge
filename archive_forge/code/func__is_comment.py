import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _is_comment(self, current_token):
    return current_token.type == TOKEN.COMMENT or current_token.type == TOKEN.BLOCK_COMMENT or current_token.type == TOKEN.UNKNOWN