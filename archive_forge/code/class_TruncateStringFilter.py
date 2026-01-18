from sqlparse import tokens as T
from sqlparse.compat import text_type
class TruncateStringFilter(object):

    def __init__(self, width, char):
        self.width = width
        self.char = char

    def process(self, stream):
        for ttype, value in stream:
            if ttype != T.Literal.String.Single:
                yield (ttype, value)
                continue
            if value[:2] == "''":
                inner = value[2:-2]
                quote = "''"
            else:
                inner = value[1:-1]
                quote = "'"
            if len(inner) > self.width:
                value = ''.join((quote, inner[:self.width], self.char, quote))
            yield (ttype, value)