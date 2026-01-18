from nltk.internals import Counter
from nltk.sem.logic import APP, LogicParser
class LinearLogicParser(LogicParser):
    """A linear logic expression parser."""

    def __init__(self):
        LogicParser.__init__(self)
        self.operator_precedence = {APP: 1, Tokens.IMP: 2, None: 3}
        self.right_associated_operations += [Tokens.IMP]

    def get_all_symbols(self):
        return Tokens.TOKENS

    def handle(self, tok, context):
        if tok not in Tokens.TOKENS:
            return self.handle_variable(tok, context)
        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def get_BooleanExpression_factory(self, tok):
        if tok == Tokens.IMP:
            return ImpExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        """Attempt to make an application expression.  If the next tokens
        are an argument in parens, then the argument expression is a
        function being applied to the arguments.  Otherwise, return the
        argument expression."""
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                self.token()
                argument = self.process_next_expression(APP)
                self.assertNextToken(Tokens.CLOSE)
                expression = ApplicationExpression(expression, argument, None)
        return expression

    def make_VariableExpression(self, name):
        if name[0].isupper():
            return VariableExpression(name)
        else:
            return ConstantExpression(name)