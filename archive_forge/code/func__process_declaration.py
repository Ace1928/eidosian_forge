import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def _process_declaration(self, declaration, tokens):
    for index, token, value in tokens:
        if self._relevant(token):
            break
        yield (index, token, value)
    if declaration == 'datatype':
        prev_was_colon = False
        token = Keyword.Type if token == Literal else token
        yield (index, token, value)
        for index, token, value in tokens:
            if prev_was_colon and token == Literal:
                token = Keyword.Type
            yield (index, token, value)
            if self._relevant(token):
                prev_was_colon = token == Literal and value == ':'
    elif declaration == 'package':
        token = Name.Namespace if token == Literal else token
        yield (index, token, value)
    elif declaration == 'define':
        token = Name.Function if token == Literal else token
        yield (index, token, value)
        for index, token, value in tokens:
            if self._relevant(token):
                break
            yield (index, token, value)
        if value == '{' and token == Literal:
            yield (index, Punctuation, value)
            for index, token, value in self._process_signature(tokens):
                yield (index, token, value)
        else:
            yield (index, token, value)
    else:
        token = Name.Function if token == Literal else token
        yield (index, token, value)
    raise StopIteration