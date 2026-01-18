import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class KeywordTable(TestCaseTable):
    _tokenizer_class = KeywordCall
    _setting_class = KeywordSetting

    def _is_template(self, value):
        return False