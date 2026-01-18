from antlr4.Recognizer import Recognizer
from antlr4.RuleContext import RuleContext
from io import StringIO
class SemanticContext(object):
    NONE = None

    def eval(self, parser: Recognizer, outerContext: RuleContext):
        pass

    def evalPrecedence(self, parser: Recognizer, outerContext: RuleContext):
        return self