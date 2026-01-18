from antlr4 import *
from io import StringIO
import sys
class SampleByBucketContext(SampleMethodContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.sampleType = None
        self.numerator = None
        self.denominator = None
        self.copyFrom(ctx)

    def OUT(self):
        return self.getToken(fugue_sqlParser.OUT, 0)

    def OF(self):
        return self.getToken(fugue_sqlParser.OF, 0)

    def BUCKET(self):
        return self.getToken(fugue_sqlParser.BUCKET, 0)

    def INTEGER_VALUE(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.INTEGER_VALUE)
        else:
            return self.getToken(fugue_sqlParser.INTEGER_VALUE, i)

    def ON(self):
        return self.getToken(fugue_sqlParser.ON, 0)

    def identifier(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierContext, 0)

    def qualifiedName(self):
        return self.getTypedRuleContext(fugue_sqlParser.QualifiedNameContext, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitSampleByBucket'):
            return visitor.visitSampleByBucket(self)
        else:
            return visitor.visitChildren(self)