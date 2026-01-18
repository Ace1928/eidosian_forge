import functools
import re
import warnings
@classmethod
def _parse_to_clause(cls, expression):
    return cls.Parser.parse(expression)