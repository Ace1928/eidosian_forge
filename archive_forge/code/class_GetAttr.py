from inspect import isclass
class GetAttr(Type):
    """
            Type of a named attribute
            """

    def __init__(self, param, attr):
        super(GetAttr, self).__init__(param=param, attr=attr)

    def generate(self, ctx):
        return 'decltype(pythonic::builtins::getattr({}{{}}, {}))'.format('pythonic::types::attr::' + self.attr.upper(), 'std::declval<' + ctx(self.param) + '>()')