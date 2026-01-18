import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_non_javascript(self, c):
    resulting_string = ''
    if c == '#':
        if self._is_first_token():
            resulting_string = self._patterns.shebang.read()
            if resulting_string:
                return self._create_token(TOKEN.UNKNOWN, resulting_string.strip() + '\n')
        resulting_string = self._patterns.include.read()
        if resulting_string:
            return self._create_token(TOKEN.UNKNOWN, resulting_string.strip() + '\n')
        c = self._input.next()
        sharp = '#'
        if self._input.hasNext() and self._input.testChar(digit):
            while True:
                c = self._input.next()
                sharp += c
                if not self._input.hasNext() or c == '#' or c == '=':
                    break
            if c == '#':
                pass
            elif self._input.peek() == '[' and self._input.peek(1) == ']':
                sharp += '[]'
                self._input.next()
                self._input.next()
            elif self._input.peek() == '{' and self._input.peek(1) == '}':
                sharp += '{}'
                self._input.next()
                self._input.next()
            return self._create_token(TOKEN.WORD, sharp)
        self._input.back()
    elif c == '<' and self._is_first_token():
        if self._patterns.html_comment_start.read():
            c = '<!--'
            while self._input.hasNext() and (not self._input.testChar(self.acorn.newline)):
                c += self._input.next()
            self.in_html_comment = True
            return self._create_token(TOKEN.COMMENT, c)
    elif c == '-' and self.in_html_comment and self._patterns.html_comment_end.read():
        self.in_html_comment = False
        return self._create_token(TOKEN.COMMENT, '-->')
    return None