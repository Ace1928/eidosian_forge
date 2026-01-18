from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
class FunctionPrototype(Type):
    """
    Represents the prototype of a first-class function type.
    Used internally.
    """
    cconv = None

    def __init__(self, rtype, atypes):
        self.rtype = rtype
        self.atypes = tuple(atypes)
        assert isinstance(rtype, Type), rtype
        lst = []
        for atype in self.atypes:
            assert isinstance(atype, Type), atype
            lst.append(atype.name)
        name = '%s(%s)' % (rtype, ', '.join(lst))
        super(FunctionPrototype, self).__init__(name)

    @property
    def key(self):
        return self.name