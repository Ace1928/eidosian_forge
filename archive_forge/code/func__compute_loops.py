from snappy.snap import t3mlite as t3m
from truncatedComplex import *
def _compute_loops(self):
    self.loops = [self._compute_loop(alpha_edge) for alpha_edge in self.alpha_edges]