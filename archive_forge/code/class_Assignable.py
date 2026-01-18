from inspect import isclass
class Assignable(DependentType):
    """
            A type which can be assigned

            It is used to make the difference between
            * transient types (e.g. generated from expression template)
            * assignable types (typically type of a variable)
            """

    def generate(self, ctx):
        return 'typename pythonic::assignable<{0}>::type'.format(ctx(self.of))