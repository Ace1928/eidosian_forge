from inspect import isclass
class PType(Type):
    """
            A generic parametric type
            """
    prefix = '__ptype{0}'
    count = 0

    def __init__(self, fun, ptype):
        super(PType, self).__init__(fun=fun, type=ptype, name=PType.prefix.format(PType.count))
        PType.count += 1

    def generate(self, ctx):
        return ctx(self.type)

    def instanciate(self, caller, arguments):
        if self.fun is caller:
            return builder.UnknownType
        else:
            return InstantiatedType(self.fun, self.name, arguments)