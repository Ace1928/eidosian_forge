from sqlparse import tokens as T
from sqlparse.compat import text_type
class IdentifierCaseFilter(_CaseFilter):
    ttype = (T.Name, T.String.Symbol)

    def process(self, stream):
        for ttype, value in stream:
            if ttype in self.ttype and value.strip()[0] != '"':
                value = self.convert(value)
            yield (ttype, value)