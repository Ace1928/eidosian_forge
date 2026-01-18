import json
from ..error import GraphQLSyntaxError
class TokenKind(object):
    EOF = 1
    BANG = 2
    DOLLAR = 3
    PAREN_L = 4
    PAREN_R = 5
    SPREAD = 6
    COLON = 7
    EQUALS = 8
    AT = 9
    BRACKET_L = 10
    BRACKET_R = 11
    BRACE_L = 12
    PIPE = 13
    BRACE_R = 14
    NAME = 15
    VARIABLE = 16
    INT = 17
    FLOAT = 18
    STRING = 19