from ..libmp.backend import xrange
from .calculus import defun

        This routine applies the convergence acceleration to the list of partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k ,k = 0..n)

        v, e = ...update_psum([s_0, s_1,..., s_k])

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        