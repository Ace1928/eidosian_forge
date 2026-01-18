from antlr4 import *
from io import StringIO
import sys
class ComplexDataTypeContext(DataTypeContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.icomplex = None
        self.copyFrom(ctx)

    def LT(self):
        return self.getToken(fugue_sqlParser.LT, 0)

    def dataType(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.DataTypeContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.DataTypeContext, i)

    def GT(self):
        return self.getToken(fugue_sqlParser.GT, 0)

    def ARRAY(self):
        return self.getToken(fugue_sqlParser.ARRAY, 0)

    def MAP(self):
        return self.getToken(fugue_sqlParser.MAP, 0)

    def STRUCT(self):
        return self.getToken(fugue_sqlParser.STRUCT, 0)

    def NEQ(self):
        return self.getToken(fugue_sqlParser.NEQ, 0)

    def complexColTypeList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ComplexColTypeListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitComplexDataType'):
            return visitor.visitComplexDataType(self)
        else:
            return visitor.visitChildren(self)