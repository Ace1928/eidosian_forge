from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _compute_fixed_pts(self):
    self.fixed_pts_for_loops = [_fixed_points(m) for m in self.matrix_for_loops]