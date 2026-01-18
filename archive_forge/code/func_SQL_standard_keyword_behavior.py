from antlr4 import *
from io import StringIO
import sys
@property
def SQL_standard_keyword_behavior(self):
    if '_ansi_sql' in self.__dict__:
        return self._ansi_sql
    return False