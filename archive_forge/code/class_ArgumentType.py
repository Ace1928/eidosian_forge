from inspect import isclass
class ArgumentType(Type):
    """
            A type to hold function arguments
            """

    def __init__(self, num):
        super(ArgumentType, self).__init__(num=num)

    def generate(self, _):
        argtype = 'argument_type{0}'.format(self.num)
        noref = 'typename std::remove_reference<{0}>::type'.format(argtype)
        return 'typename std::remove_cv<{0}>::type'.format(noref)