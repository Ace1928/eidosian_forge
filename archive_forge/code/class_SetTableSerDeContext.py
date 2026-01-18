from antlr4 import *
from io import StringIO
import sys
class SetTableSerDeContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.copyFrom(ctx)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def TABLE(self):
        return self.getToken(fugue_sqlParser.TABLE, 0)

    def multipartIdentifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.MultipartIdentifierContext, 0)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def SERDE(self):
        return self.getToken(fugue_sqlParser.SERDE, 0)

    def STRING(self):
        return self.getToken(fugue_sqlParser.STRING, 0)

    def partitionSpec(self):
        return self.getTypedRuleContext(fugue_sqlParser.PartitionSpecContext, 0)

    def WITH(self):
        return self.getToken(fugue_sqlParser.WITH, 0)

    def SERDEPROPERTIES(self):
        return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

    def tablePropertyList(self):
        return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSetTableSerDe'):
            return visitor.visitSetTableSerDe(self)
        else:
            return visitor.visitChildren(self)