from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
class BarycentricArc(Arc):
    """
    A line segment between two endpoints in barycentric coordinates.
    """

    def __init__(self, start, end, past=None, next=None, tet=None):
        self.start = start
        self.end = end
        self.past = past
        self.next = next
        self.tet = tet

    def __hash__(self):
        return hash((self.start, self.end, self.tet))

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end and (self.tet == other.tet)

    def trim(self):
        """
        Given an arc in R^4 lying in the slice x0 + x1 + x2 + x3 = 1,
        return its intersection with the standard three-simplex.
        """
        t0, t1 = (0, 1)
        u, v = (self.start.vector, self.end.vector)
        for i in range(4):
            if u[i] < 0 and v[i] < 0:
                return None
            elif u[i] >= 0 and v[i] >= 0:
                continue
            else:
                t = u[i] / (u[i] - v[i])
                assert (1 - t) * u[i] + t * v[i] == 0
                if u[i] < 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
        if t1 < t0:
            return None
        x = (1 - t0) * u + t0 * v
        y = (1 - t1) * u + t1 * v
        return BarycentricArc(BarycentricPoint(*x), BarycentricPoint(*y))

    def __repr__(self):
        return '[{},{}]'.format(self.start, self.end)

    def transform_to_R3(self, matrix, bdry_map=None):
        new_start = self.start.transform_to_R3(matrix, bdry_map)
        new_end = self.end.transform_to_R3(matrix, bdry_map)
        return BarycentricArc(new_start, new_end)

    def is_nongeneric(self):
        zeros_s = self.start.zero_coordinates()
        zeros_e = self.end.zero_coordinates()
        if len(zeros_s) > 1 or len(zeros_e) > 1:
            return True
        if len(set(zeros_s) & set(zeros_e)) > 0:
            return True
        return False

    def max_denom(self):
        rationals = list(self.start.vector) + list(self.end.vector)
        return max((q.denominator() for q in rationals))