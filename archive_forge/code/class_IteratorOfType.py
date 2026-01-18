from inspect import isclass
class IteratorOfType(DependentType):
    """
            Type of an Iterator of a container
            """

    def generate(self, ctx):
        container_type = ctx(self.of)
        if container_type.startswith('typename'):
            return container_type + '::iterator'
        else:
            return 'typename ' + container_type + '::iterator'