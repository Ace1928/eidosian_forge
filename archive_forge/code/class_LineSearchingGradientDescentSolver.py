from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
class LineSearchingGradientDescentSolver(SolverBase):

    def step(self, x, iter_idx, maxiter):
        return self.line_search(x, self._gd_step(x))