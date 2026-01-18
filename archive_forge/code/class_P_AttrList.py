from pyparsing import (
import pydot
class P_AttrList(object):

    def __init__(self, toks):
        self.attrs = {}
        i = 0
        while i < len(toks):
            attrname = toks[i]
            if i + 2 < len(toks) and toks[i + 1] == '=':
                attrvalue = toks[i + 2]
                i += 3
            else:
                attrvalue = None
                i += 1
            self.attrs[attrname] = attrvalue

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.attrs)