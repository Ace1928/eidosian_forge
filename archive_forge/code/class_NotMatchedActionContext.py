from antlr4 import *
from io import StringIO
import sys
class NotMatchedActionContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.columns = None

    def INSERT(self):
        return self.getToken(fugue_sqlParser.INSERT, 0)

    def ASTERISK(self):
        return self.getToken(fugue_sqlParser.ASTERISK, 0)

    def VALUES(self):
        return self.getToken(fugue_sqlParser.VALUES, 0)

    def expression(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ExpressionContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ExpressionContext, i)

    def multipartIdentifierList(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_notMatchedAction

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitNotMatchedAction'):
            return visitor.visitNotMatchedAction(self)
        else:
            return visitor.visitChildren(self)