from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
class SerializerUnicode(object):

    @staticmethod
    def process(stmt):
        lines = split_unquoted_newlines(stmt)
        return '\n'.join((line.rstrip() for line in lines))