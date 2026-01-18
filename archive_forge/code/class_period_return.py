import sys
from antlr3 import *
from antlr3.compat import set, frozenset
class period_return(ParserRuleReturnScope):

    def __init__(self):
        ParserRuleReturnScope.__init__(self)