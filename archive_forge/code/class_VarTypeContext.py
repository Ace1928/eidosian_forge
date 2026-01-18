from antlr4 import *
from io import StringIO
import sys
class VarTypeContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def Newtonian(self):
        return self.getToken(AutolevParser.Newtonian, 0)

    def Frames(self):
        return self.getToken(AutolevParser.Frames, 0)

    def Bodies(self):
        return self.getToken(AutolevParser.Bodies, 0)

    def Particles(self):
        return self.getToken(AutolevParser.Particles, 0)

    def Points(self):
        return self.getToken(AutolevParser.Points, 0)

    def Constants(self):
        return self.getToken(AutolevParser.Constants, 0)

    def Specifieds(self):
        return self.getToken(AutolevParser.Specifieds, 0)

    def Imaginary(self):
        return self.getToken(AutolevParser.Imaginary, 0)

    def Variables(self):
        return self.getToken(AutolevParser.Variables, 0)

    def MotionVariables(self):
        return self.getToken(AutolevParser.MotionVariables, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_varType

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterVarType'):
            listener.enterVarType(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitVarType'):
            listener.exitVarType(self)