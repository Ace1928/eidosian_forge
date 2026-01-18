from inspect import isclass
class AssignableNoEscape(DependentType):
    """
            Similar to Assignable, but it doesn't escape it's declaration scope
            """

    def generate(self, ctx):
        return 'typename pythonic::assignable_noescape<{0}>::type'.format(ctx(self.of))