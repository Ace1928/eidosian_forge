from ...sage_helper import _within_sage
from ..upper_halfspace.finite_point import *
def _canonical_translates(self, z):

    def round_to_nearest_integers(interval):
        r = interval.round()
        return list(range(int(r.lower()), int(r.upper()) + 1))
    v = self._to_vec(z)
    integer_ranges = [round_to_nearest_integers(i) for i in v]
    for i in integer_ranges[0]:
        for j in integer_ranges[1]:
            yield (z - i * self.t0 - j * self.t1)