from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def addMeta(s, parent, lev):
    misc = E.Element('miscellaneous')
    mf = 0
    for mtype, mval in sorted(s.metadata.items()):
        if mtype == 'S':
            addElemT(parent, 'source', mval, lev)
        elif mtype in s.metaTypes:
            continue
        else:
            mf = E.Element('miscellaneous-field', name=s.metaTab[mtype])
            mf.text = mval
            addElem(misc, mf, lev + 1)
    if mf != 0:
        addElem(parent, misc, lev)