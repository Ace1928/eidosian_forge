from pyparsing import *
def addDigit(n, limit, c, s):
    n -= limit
    s += c
    return (n, s)