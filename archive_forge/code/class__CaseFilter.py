from sqlparse import tokens as T
from sqlparse.compat import text_type
class _CaseFilter(object):
    ttype = None

    def __init__(self, case=None):
        case = case or 'upper'
        self.convert = getattr(text_type, case)

    def process(self, stream):
        for ttype, value in stream:
            if ttype in self.ttype:
                value = self.convert(value)
            yield (ttype, value)