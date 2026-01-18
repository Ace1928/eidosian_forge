import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class TestCaseTable(_Table):
    _setting_class = TestCaseSetting
    _test_template = None
    _default_template = None

    @property
    def _tokenizer_class(self):
        if self._test_template or (self._default_template and self._test_template is not False):
            return TemplatedKeywordCall
        return KeywordCall

    def _continues(self, value, index):
        return index > 0 and _Table._continues(self, value, index)

    def _tokenize(self, value, index):
        if index == 0:
            if value:
                self._test_template = None
            return GherkinTokenizer().tokenize(value, TC_KW_NAME)
        if index == 1 and self._is_setting(value):
            if self._is_template(value):
                self._test_template = False
                self._tokenizer = self._setting_class(self.set_test_template)
            else:
                self._tokenizer = self._setting_class()
        if index == 1 and self._is_for_loop(value):
            self._tokenizer = ForLoop()
        if index == 1 and self._is_empty(value):
            return [(value, SYNTAX)]
        return _Table._tokenize(self, value, index)

    def _is_setting(self, value):
        return value.startswith('[') and value.endswith(']')

    def _is_template(self, value):
        return normalize(value) == '[template]'

    def _is_for_loop(self, value):
        return value.startswith(':') and normalize(value, remove=':') == 'for'

    def set_test_template(self, template):
        self._test_template = self._is_template_set(template)

    def set_default_template(self, template):
        self._default_template = self._is_template_set(template)

    def _is_template_set(self, template):
        return normalize(template) not in ('', '\\', 'none', '${empty}')