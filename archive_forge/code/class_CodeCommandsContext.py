from antlr4 import *
from io import StringIO
import sys
class CodeCommandsContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def units(self):
        return self.getTypedRuleContext(AutolevParser.UnitsContext, 0)

    def inputs(self):
        return self.getTypedRuleContext(AutolevParser.InputsContext, 0)

    def outputs(self):
        return self.getTypedRuleContext(AutolevParser.OutputsContext, 0)

    def codegen(self):
        return self.getTypedRuleContext(AutolevParser.CodegenContext, 0)

    def commands(self):
        return self.getTypedRuleContext(AutolevParser.CommandsContext, 0)

    def getRuleIndex(self):
        return AutolevParser.RULE_codeCommands

    def enterRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'enterCodeCommands'):
            listener.enterCodeCommands(self)

    def exitRule(self, listener: ParseTreeListener):
        if hasattr(listener, 'exitCodeCommands'):
            listener.exitCodeCommands(self)