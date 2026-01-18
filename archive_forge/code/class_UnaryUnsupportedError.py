import re,sys
from pyparsing import Word, alphas, ParseException, Literal, CaselessLiteral \
class UnaryUnsupportedError(Exception):
    pass