from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def addElem(parent, child, level):
    indent = 2
    chldrn = list(parent)
    if chldrn:
        chldrn[-1].tail += indent * ' '
    else:
        parent.text = '\n' + level * indent * ' '
    parent.append(child)
    child.tail = '\n' + (level - 1) * indent * ' '