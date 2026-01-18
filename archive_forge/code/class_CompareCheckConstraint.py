from itertools import zip_longest
from sqlalchemy import schema
from sqlalchemy.sql.elements import ClauseList
class CompareCheckConstraint:

    def __init__(self, constraint):
        self.constraint = constraint

    def __eq__(self, other):
        return isinstance(other, schema.CheckConstraint) and self.constraint.name == other.name and (str(self.constraint.sqltext) == str(other.sqltext)) and (other.table.name == self.constraint.table.name) and (other.table.schema == self.constraint.table.schema)

    def __ne__(self, other):
        return not self.__eq__(other)