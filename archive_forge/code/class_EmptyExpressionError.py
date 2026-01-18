from jmespath.compat import with_str_method
class EmptyExpressionError(JMESPathError):

    def __init__(self):
        super(EmptyExpressionError, self).__init__('Invalid JMESPath expression: cannot be empty.')