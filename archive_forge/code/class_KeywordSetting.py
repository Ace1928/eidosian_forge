import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class KeywordSetting(TestCaseSetting):
    _keyword_settings = ('teardown',)
    _other_settings = ('documentation', 'arguments', 'return', 'timeout', 'tags')