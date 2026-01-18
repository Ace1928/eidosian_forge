from pyparsing import (
import pydot
class DefaultStatement(P_AttrList):

    def __init__(self, default_type, attrs):
        self.default_type = default_type
        self.attrs = attrs

    def __repr__(self):
        return '%s(%s, %r)' % (self.__class__.__name__, self.default_type, self.attrs)