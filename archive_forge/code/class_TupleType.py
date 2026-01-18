from inspect import isclass
class TupleType(Type):
    """
            Type holding a tuple of stuffs of various types
            """

    def __init__(self, *ofs):
        super(TupleType, self).__init__(ofs=ofs)

    def iscombined(self):
        return any((of.iscombined() for of in self.ofs))

    def generate(self, ctx):
        elts = (ctx(of) for of in self.ofs)
        telts = ('std::declval<{0}>()'.format(elt) for elt in elts)
        return 'decltype(pythonic::types::make_tuple({0}))'.format(', '.join(telts))