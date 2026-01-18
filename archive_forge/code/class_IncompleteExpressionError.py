from jmespath.compat import with_str_method
@with_str_method
class IncompleteExpressionError(ParseError):

    def set_expression(self, expression):
        self.expression = expression
        self.lex_position = len(expression)
        self.token_type = None
        self.token_value = None

    def __str__(self):
        underline = ' ' * (self.lex_position + 1) + '^'
        return 'Invalid jmespath expression: Incomplete expression:\n"%s"\n%s' % (self.expression, underline)