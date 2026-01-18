import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class ContainerVSpace(VSpace):

    def __init__(self, value):
        self.shape = value
        self.shape = self._map(vspace)

    @property
    def size(self):
        return sum(self._values(self._map(lambda vs: vs.size)))

    def zeros(self):
        return self._map(lambda vs: vs.zeros())

    def ones(self):
        return self._map(lambda vs: vs.ones())

    def randn(self):
        return self._map(lambda vs: vs.randn())

    def standard_basis(self):
        zero = self.zeros()
        for i, vs in self._kv_pairs(self.shape):
            for x in vs.standard_basis():
                yield self._subval(zero, i, x)

    def _add(self, xs, ys):
        return self._map(lambda vs, x, y: vs._add(x, y), xs, ys)

    def _mut_add(self, xs, ys):
        return self._map(lambda vs, x, y: vs._mut_add(x, y), xs, ys)

    def _scalar_mul(self, xs, a):
        return self._map(lambda vs, x: vs._scalar_mul(x, a), xs)

    def _inner_prod(self, xs, ys):
        return sum(self._values(self._map(lambda vs, x, y: vs._inner_prod(x, y), xs, ys)))

    def _covector(self, xs):
        return self._map(lambda vs, x: vs._covector(x), xs)