from itertools import zip_longest
from sqlalchemy import schema
from sqlalchemy.sql.elements import ClauseList
class CompareColumn:

    def __init__(self, column):
        self.column = column

    def __eq__(self, other):
        return self.column.name == other.name and self.column.nullable == other.nullable

    def __ne__(self, other):
        return not self.__eq__(other)