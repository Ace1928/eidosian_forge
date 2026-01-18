from ..sage_helper import _within_sage
from snappy.number import SnapPyNumbers, Number
from itertools import chain
from ..pari import pari, PariError
from .fundamental_polyhedron import Infinity
class Matrix2x2(MatrixBase):
    """A 2x2 matrix class whose entries are snappy Numbers."""

    def __init__(self, *args):
        if is_field(args[0]):
            self._base_ring = number = SnapPyNumbers(args[0].precision())
            args = args[1:]
        else:
            self._base_ring = None
            number = Number
        if len(args) == 1:
            args = tuple(chain(*args[0]))
        if len(args) == 4:
            self.a, self.b, self.c, self.d = [number(x) for x in args]
        else:
            raise ValueError('Invalid initialization for a Matrix2x2.')

    def __repr__(self):
        entries = [str(e) for e in self.list()]
        size = max(map(len, entries))
        entries = tuple(('%%-%d.%ds' % (size, size) % x for x in entries))
        return '[ %s  %s ]\n[ %s  %s ]' % entries

    def __getitem__(self, index):
        if isinstance(index, int):
            if index == 0:
                return [self.a, self.b]
            if index == 1:
                return [self.c, self.d]
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index
            if i == 0:
                return self.a if j == 0 else self.b
            if i == 1:
                return self.c if j == 0 else self.d
        raise IndexError('Invalid 2x2 matrix index.')

    def __add__(self, other):
        return Matrix2x2(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    def __sub__(self, other):
        return Matrix2x2(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    def __mul__(self, other):
        if isinstance(other, Matrix2x2):
            return Matrix2x2(self.a * other.a + self.b * other.c, self.a * other.b + self.b * other.d, self.c * other.a + self.d * other.c, self.c * other.b + self.d * other.d)
        if isinstance(other, Vector2):
            return Vector2(self.a * other.x + self.b * other.y, self.c * other.x + self.d * other.y)
        if isinstance(other, Number):
            return Matrix2x2(self.a * other, self.b * other, self.c * other, self.d * other)
        try:
            return self * self.base_ring()(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __rmul__(self, other):
        return Matrix2x2(self.a * other, self.b * other, self.c * other, self.d * other)

    def __div__(self, other):
        return Matrix2x2(self.a / other, self.b / other, self.c / other, self.d / other)

    def __truediv__(self, other):
        return Matrix2x2(self.a / other, self.b / other, self.c / other, self.d / other)

    def __neg__(self):
        return Matrix2x2(-self.a, -self.b, -self.c, -self.d)

    def __invert__(self):
        try:
            D = 1 / self.det()
        except ZeroDivisionError:
            raise ZeroDivisionError('matrix %s is not invertible.' % self)
        return Matrix2x2(self.d * D, -self.b * D, -self.c * D, self.a * D)

    def adjoint(self):
        return Matrix2x2(self.d, -self.b, -self.c, self.a)

    def determinant(self):
        return self.a * self.d - self.b * self.c
    det = determinant

    def trace(self):
        return self.a + self.d

    def eigenvalues(self):
        R = self.base_ring()
        x = pari('x')
        a, b, c, d = map(pari, self.list())
        p = x * x - (a + d) * x + (a * d - b * c)
        roots = p.polroots(precision=R.precision())
        return map(R, roots)

    def norm(self, p=2):
        if p == 1:
            return max(self.a.abs() + self.c.abs(), self.b.abs() + self.d.abs())
        if p == 'frob':
            return sum([x * x for x in self.list()]).sqrt()
        if p == Infinity:
            return max(self.a.abs() + self.b.abs(), self.c.abs() + self.d.abs())
        if p == 2:
            return max([x.abs() for x in self.eigenvalues()])

    def list(self):
        return [self.a, self.b, self.c, self.d]

    def rows(self):
        return [Vector2(self.base_ring(), self.a, self.b), Vector2(self.base_ring(), self.a, self.b)]

    def sage(self):
        return sage_matrix(2, 2, [x.sage() for x in self.list()])