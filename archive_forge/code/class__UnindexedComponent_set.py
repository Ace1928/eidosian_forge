from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.range import NonNumericRange
class _UnindexedComponent_set(GlobalSetBase):
    local_name = 'UnindexedComponent_set'
    _anonymous_sets = GlobalSetBase

    def __init__(self, name):
        self.name = name
        self._constructed = True

    def __contains__(self, val):
        return val is None

    def get(self, value, default):
        if value is None:
            return value
        return default

    def __iter__(self):
        return (None,).__iter__()

    def __reversed__(self):
        return iter(self)

    def ordered_iter(self):
        return iter(self)

    def sorted_iter(self):
        return iter(self)

    def data(self):
        return tuple(self)

    def ordered_data(self):
        return tuple(self)

    def sorted_data(self):
        return tuple(self)

    def subsets(self, expand_all_set_operators=None):
        return [self]

    def construct(self):
        pass

    def ranges(self):
        yield NonNumericRange(None)

    def bounds(self):
        return (None, None)

    def get_interval(self):
        return (None, None, None)

    def __len__(self):
        return 1

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    @property
    def dimen(self):
        return 0

    def isdiscrete(self):
        return True

    def isfinite(self):
        return True

    def isordered(self):
        return True

    def at(self, index):
        if index == 1:
            return None
        raise IndexError('%s index out of range' % (self.name,))

    def ord(self, item):
        if item is None:
            return 1
        raise IndexError('Cannot identify position of %s in Set %s: item not in Set' % (item, self.name))

    def first(self):
        return None

    def last(self):
        return None

    def next(self, item, step=1):
        self.ord(item)
        if step < 0:
            raise IndexError('Cannot advance before the beginning of the Set')
        else:
            raise IndexError('Cannot advance past the end of the Set')

    def nextw(self, item, step=1):
        self.ord(item)
        return None

    def prev(self, item, step=1):
        return self.next(item, -step)

    def prevw(self, item, step=1):
        return self.nextw(item, -step)

    def parent_block(self):
        return None

    def parent_component(self):
        return self