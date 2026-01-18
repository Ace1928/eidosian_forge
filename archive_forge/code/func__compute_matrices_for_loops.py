from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _compute_matrices_for_loops(self):

    def _to_psl(m):
        return m / m.determinant().sqrt()
    self.matrix_for_loops = [_to_psl(self.hyperbolic_structure.pgl2_matrix_for_path(loop)) for loop in self.loops]