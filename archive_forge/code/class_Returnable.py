from inspect import isclass
class Returnable(DependentType):
    """
            A type which can be returned

            It is used to make the difference between
            * returned types (that cannot hold a reference to avoid dangling
                              reference)
            * assignable types (local to a function)

            """

    def generate(self, ctx):
        return 'typename pythonic::returnable<{0}>::type'.format(ctx(self.of))