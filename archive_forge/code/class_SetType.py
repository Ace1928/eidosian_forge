from inspect import isclass
class SetType(DependentType):
    """
            Type holding a set of stuff of the same type
            """

    def generate(self, ctx):
        return 'pythonic::types::set<{0}>'.format(ctx(self.of))