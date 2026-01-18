from inspect import isclass
class AddConst(DependentType):
    """
            Type of an Iterator of a container
            """

    def generate(self, ctx):
        of_type = ctx(self.of)
        return 'decltype(pythonic::types::as_const(std::declval<' + of_type + '>()))'