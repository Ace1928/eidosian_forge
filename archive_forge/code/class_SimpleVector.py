from .sage_helper import _within_sage
from . import number
from .math_basics import is_Interval
class SimpleVector(number.SupportsMultiplicationByNumber):

    def __init__(self, list_of_values):
        self.data = list_of_values
        try:
            self.type = type(self.data[0])
            self.shape = (len(list_of_values),)
        except IndexError:
            self.type = type(0)
            self.shape = (0,)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __repr__(self):
        str_vector = [str(x) for x in self.data]
        size = max((len(x) for x in str_vector))
        return '(%s)' % ', '.join(('% *s' % (size, x) for x in str_vector))

    def __getitem__(self, key):
        if key < 0:
            raise TypeError("Simple vectors don't have negative indices.")
        return self.data[key]

    def __setitem__(self, key, value):
        if key < 0:
            raise TypeError("Simple vectors don't have negative indices.")
        self.data[key] = value

    def entries(self):
        return [x for x in self.data]

    def list(self):
        return self.entries()

    def normalized(self):
        l = sum([abs(x) ** 2 for x in self.data]).sqrt()
        return SimpleVector([x / l for x in self.data])

    def __add__(self, other):
        if isinstance(other, SimpleVector):
            if self.shape[0] != other.shape[0]:
                raise ValueError('Cannot add vector of length %d and vector of length %d.' % (self.shape[0], other.shape[0]))
            return SimpleVector([a + b for a, b in zip(self.data, other.data)])
        return ValueError('SimpleVector only supports addition for another SimpleVector. Given type was %r.' % type(other))

    def __sub__(self, other):
        if isinstance(other, SimpleVector):
            if self.shape[0] != other.shape[0]:
                raise ValueError('Cannot add vector of length %d and vector of length %d.' % (self.shape[0], other.shape[0]))
            return SimpleVector([a - b for a, b in zip(self.data, other.data)])
        return ValueError('SimpleVector only supports addition for another SimpleVector. Given type was %r.' % type(other))

    def _multiply_by_scalar(self, other):
        return SimpleVector([other * e for e in self.data])

    def __truediv__(self, other):
        return SimpleVector([x / other for x in self.data])

    def base_ring(self):
        try:
            return self.data[0].parent()
        except IndexError:
            return self.type