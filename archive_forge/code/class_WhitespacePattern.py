import re
from ..core.pattern import Pattern
class WhitespacePattern(Pattern):

    def __init__(self, input_scanner, parent=None):
        Pattern.__init__(self, input_scanner, parent)
        if parent is not None:
            self._newline_regexp = self._input.get_regexp(parent._newline_regexp)
        else:
            self.__set_whitespace_patterns('', '')
        self.newline_count = 0
        self.whitespace_before_token = ''

    def __set_whitespace_patterns(self, whitespace_chars, newline_chars):
        whitespace_chars += '\\t '
        newline_chars += '\\n\\r'
        self._match_pattern = self._input.get_regexp('[' + whitespace_chars + newline_chars + ']+')
        self._newline_regexp = self._input.get_regexp('\\r\\n|[' + newline_chars + ']')

    def read(self):
        self.newline_count = 0
        self.whitespace_before_token = ''
        resulting_string = self._input.read(self._match_pattern)
        if resulting_string == ' ':
            self.whitespace_before_token = ' '
        elif bool(resulting_string):
            lines = self._newline_regexp.split(resulting_string)
            self.newline_count = len(lines) - 1
            self.whitespace_before_token = lines[-1]
        return resulting_string

    def matching(self, whitespace_chars, newline_chars):
        result = self._create()
        result.__set_whitespace_patterns(whitespace_chars, newline_chars)
        result._update()
        return result

    def _create(self):
        return WhitespacePattern(self._input, self)