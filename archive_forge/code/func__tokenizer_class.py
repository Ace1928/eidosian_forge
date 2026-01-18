import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
@property
def _tokenizer_class(self):
    if self._test_template or (self._default_template and self._test_template is not False):
        return TemplatedKeywordCall
    return KeywordCall