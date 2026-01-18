import builtins
import inspect
import io
import keyword
import linecache
import os
import re
import sys
import sysconfig
import tokenize
import traceback
class SyntaxHighlighter:
    _default_style = {'comment': '\x1b[30m\x1b[1m{}\x1b[0m', 'keyword': '\x1b[35m\x1b[1m{}\x1b[0m', 'builtin': '\x1b[1m{}\x1b[0m', 'string': '\x1b[36m{}\x1b[0m', 'number': '\x1b[34m\x1b[1m{}\x1b[0m', 'operator': '\x1b[35m\x1b[1m{}\x1b[0m', 'punctuation': '\x1b[1m{}\x1b[0m', 'constant': '\x1b[36m\x1b[1m{}\x1b[0m', 'identifier': '\x1b[1m{}\x1b[0m', 'other': '{}'}
    _builtins = set(dir(builtins))
    _constants = {'True', 'False', 'None'}
    _punctation = {'(', ')', '[', ']', '{', '}', ':', ',', ';'}
    _strings = {tokenize.STRING}
    _fstring_middle = None
    if sys.version_info >= (3, 12):
        _strings.update({tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END})
        _fstring_middle = tokenize.FSTRING_MIDDLE

    def __init__(self, style=None):
        self._style = style or self._default_style

    def highlight(self, source):
        style = self._style
        row, column = (0, 0)
        output = ''
        for token in self.tokenize(source):
            type_, string, (start_row, start_column), (_, end_column), line = token
            if type_ == self._fstring_middle:
                end_column += string.count('{') + string.count('}')
            if type_ == tokenize.NAME:
                if string in self._constants:
                    color = style['constant']
                elif keyword.iskeyword(string):
                    color = style['keyword']
                elif string in self._builtins:
                    color = style['builtin']
                else:
                    color = style['identifier']
            elif type_ == tokenize.OP:
                if string in self._punctation:
                    color = style['punctuation']
                else:
                    color = style['operator']
            elif type_ == tokenize.NUMBER:
                color = style['number']
            elif type_ in self._strings:
                color = style['string']
            elif type_ == tokenize.COMMENT:
                color = style['comment']
            else:
                color = style['other']
            if start_row != row:
                source = source[column:]
                row, column = (start_row, 0)
            if type_ != tokenize.ENCODING:
                output += line[column:start_column]
                output += color.format(line[start_column:end_column])
            column = end_column
        output += source[column:]
        return output

    @staticmethod
    def tokenize(source):
        source = source.encode('utf-8')
        source = io.BytesIO(source)
        try:
            yield from tokenize.tokenize(source.readline)
        except tokenize.TokenError:
            return